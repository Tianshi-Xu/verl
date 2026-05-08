# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def _build_topk_kl_compute_mask(data, target_shape: torch.Size) -> torch.Tensor:
    """Build a full-sequence logits-position mask for response_mask=1 tokens."""
    input_ids = data["input_ids"]
    response_mask = data["response_mask"]
    prompts = data["prompts"]
    responses = data["responses"]

    if not input_ids.is_nested:
        raise NotImplementedError("use_response_mask_for_topk_kl requires remove-padding nested input_ids.")

    device = input_ids.values().device
    offsets = input_ids.offsets().to(device)
    if prompts.is_nested:
        prompt_lens = prompts.offsets().diff().to(device)
        response_lens = responses.offsets().diff().to(device)
    else:
        attention_mask = data["attention_mask"]
        prompt_lens = attention_mask[:, : prompts.shape[1]].sum(dim=1).to(device)
        response_lens = attention_mask[:, prompts.shape[1] :].sum(dim=1).to(device)

    response_mask_rows = response_mask.unbind() if response_mask.is_nested else response_mask.to(device).unbind(0)

    selected_indices = []
    for i, (prompt_len, response_len, response_mask_row) in enumerate(
        zip(prompt_lens.tolist(), response_lens.tolist(), response_mask_rows, strict=True)
    ):
        assert prompt_len > 0, "top-k KL response-mask optimization assumes prompt_len > 0."
        response_len = int(response_len)
        if response_len <= 0:
            continue
        token_mask = response_mask_row[:response_len].bool()
        if not token_mask.any():
            continue
        # Model output at position prompt_len - 1 + j predicts response token j.
        local_positions = int(prompt_len) - 1 + torch.arange(response_len, device=device)[token_mask]
        selected_indices.append(offsets[i] + local_positions)

    compute_mask = torch.zeros(target_shape, dtype=torch.bool, device=device)
    if selected_indices:
        selected_indices = torch.cat(selected_indices, dim=0)
        compute_mask.view(-1)[selected_indices] = True
    if get_ulysses_sequence_parallel_world_size() > 1:
        compute_mask = slice_input_tensor(compute_mask, dim=1)
    return compute_mask


def _scatter_selected_outputs(
    outputs: dict[str, torch.Tensor],
    selected_mask: torch.Tensor,
    target_shape: torch.Size,
) -> dict[str, torch.Tensor]:
    full_outputs = {}
    selected_indices = selected_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
    for key, value in outputs.items():
        full_value = value.new_zeros(target_shape)
        full_outputs[key] = full_value.view(-1).index_copy(0, selected_indices, value.view(-1)).view(target_shape)
    return full_outputs


def compute_topk_monitoring_metrics(
    student_logits: torch.Tensor,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute token-level diagnostics on the teacher top-k support."""
    with torch.no_grad():
        student_topk_probs = student_topk_log_probs.float().exp()
        teacher_topk_probs = teacher_topk_log_probs.float().exp()
        student_mass = student_topk_probs.sum(dim=-1)
        teacher_mass = teacher_topk_probs.sum(dim=-1)

        eps = torch.finfo(student_topk_probs.dtype).eps
        student_topk_norm = student_topk_probs / student_mass.clamp_min(eps).unsqueeze(-1)
        teacher_topk_norm = teacher_topk_probs / teacher_mass.clamp_min(eps).unsqueeze(-1)

        student_argmax_ids = student_logits.argmax(dim=-1)
        teacher_top1_ids = teacher_topk_ids[..., 0]
        return {
            "topk_overlap": torch.minimum(student_topk_norm, teacher_topk_norm).sum(dim=-1),
            "topk_l1": (student_topk_norm - teacher_topk_norm).abs().sum(dim=-1),
            "student_teacher_top1_prob": student_topk_probs[..., 0],
            "teacher_top1_prob": teacher_topk_probs[..., 0],
            "teacher_top1_prob_gap": teacher_topk_probs[..., 0] - student_topk_probs[..., 0],
            "student_argmax_teacher_top1_match": (student_argmax_ids == teacher_top1_ids).float(),
            "student_argmax_in_teacher_topk": (student_argmax_ids.unsqueeze(-1) == teacher_topk_ids).any(dim=-1).float(),
        }


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    data=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    loss_config: DistillationLossConfig = config.distillation_loss
    target_shape = student_logits.shape[:2]
    selected_mask = None
    if loss_config.use_response_mask_for_topk_kl:
        if data is None:
            raise ValueError("data must be provided when use_response_mask_for_topk_kl=True.")
        selected_mask = _build_topk_kl_compute_mask(data, target_shape)
        student_logits = student_logits[selected_mask].unsqueeze(0)
        teacher_topk_log_probs = teacher_topk_log_probs[selected_mask].unsqueeze(0)
        teacher_topk_ids = teacher_topk_ids[selected_mask].unsqueeze(0)

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    metrics = compute_topk_monitoring_metrics(
        student_logits=student_logits,
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
        teacher_topk_ids=teacher_topk_ids,
    )
    student_mass = student_topk_log_probs.detach().exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.detach().exp().sum(dim=-1)
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    outputs = {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        **metrics,
    }
    if selected_mask is not None:
        outputs = _scatter_selected_outputs(outputs, selected_mask, target_shape)
    return outputs


def compute_backward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    data=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute backward KL distillation loss on normalized teacher top-k support.

    This computes KL(student_topk_norm || teacher_topk_norm), where both distributions
    are renormalized over the teacher top-k tokens available from the teacher server.
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    loss_config: DistillationLossConfig = config.distillation_loss
    target_shape = student_logits.shape[:2]
    selected_mask = None
    if loss_config.use_response_mask_for_topk_kl:
        if data is None:
            raise ValueError("data must be provided when use_response_mask_for_topk_kl=True.")
        selected_mask = _build_topk_kl_compute_mask(data, target_shape)
        student_logits = student_logits[selected_mask].unsqueeze(0)
        teacher_topk_log_probs = teacher_topk_log_probs[selected_mask].unsqueeze(0)
        teacher_topk_ids = teacher_topk_ids[selected_mask].unsqueeze(0)

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    metrics = compute_topk_monitoring_metrics(
        student_logits=student_logits,
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
        teacher_topk_ids=teacher_topk_ids,
    )
    student_mass = student_topk_log_probs.detach().exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.detach().exp().sum(dim=-1)
    if loss_config.log_prob_min_clamp is not None:
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    student_topk_log_probs = student_topk_log_probs - torch.logsumexp(student_topk_log_probs, dim=-1, keepdim=True)
    teacher_topk_log_probs = teacher_topk_log_probs - torch.logsumexp(teacher_topk_log_probs, dim=-1, keepdim=True)
    distillation_losses = kl_divergence(log_q=teacher_topk_log_probs, log_p=student_topk_log_probs)

    outputs = {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        **metrics,
    }
    if selected_mask is not None:
        outputs = _scatter_selected_outputs(outputs, selected_mask, target_shape)
    return outputs
