# Porting verl/vLLM To PPU

This note is a task-independent checklist for adapting a project-bundled verl to
the local PPU environment. It is written for the case where another repository
already contains its own verl copy, for example verl 0.6.x, and that copy has
not yet been patched for PPU.

The goal is not to prescribe one project layout. The goal is to identify the
environment settings and verl/vLLM code paths that usually need to be changed so
a training run can start and pass rollout weight synchronization on PPU.

## Assumptions

- The runtime may use the system Python from the image rather than a local `uv`
  virtual environment.
- The project may contain an older verl, such as 0.6.1. File names can differ
  from the latest verl, so search by function/class names when paths do not
  match exactly.
- This PPU build presents itself through CUDA-compatible APIs:
  `torch.cuda.is_available()` can return `True`, and Ray scheduling usually uses
  the `"GPU"` resource name.
- Use the platform-provided PPU Python packages. Do not replace torch, vLLM, Ray,
  or PPU extension packages with public PyPI wheels unless you are rebuilding the
  whole runtime intentionally.

## Minimal Environment

Before changing code, verify the runtime Python sees the PPU torch/vLLM stack:

```bash
which python
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as e:
    print("vllm import failed:", repr(e))
try:
    import ray
    print("ray", ray.__version__)
except Exception as e:
    print("ray import failed:", repr(e))
PY
```

Known-good reference versions from the current image:

```text
torch:        2.9.0
vllm:         0.18.0
ray:          2.55.1
transformers: 4.57.6
flash_attn:   2.8.2
acext:        installed
```

Exact versions may differ in another image, but torch/vLLM/Ray must come from
the PPU-compatible environment.

## pip Source

Use the system/default PPU pip source for PPU packages:

```bash
export PIP_INDEX_URL=https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/ubuntu_cu129/simple/
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_ROOT_USER_ACTION=ignore
```

If the project uses system Python, install with that Python:

```bash
python -m pip install <package>
```

If the project uses `uv`, keep the same pip source and install into the active
environment:

```bash
uv pip install <package>
```

Do not globally set `TMPDIR=/mnt/data/xts/tmp`. Some runtime libraries create
temporary execution/build directories and expect normal cleanup semantics. If a
single install command needs persistent temp space, scope it only to that
command:

```bash
TMPDIR=/mnt/data/xts/tmp python -m pip install <package>
```

For runtime jobs, prefer:

```bash
export TMPDIR="${RUNTIME_TMPDIR:-/tmp/xts-runtime-tmp}"
mkdir -p "$TMPDIR"
```

## Required Runtime Variables

Set these before launching verl:

```bash
export HYDRA_FULL_ERROR=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export RAY_DISABLE_GPU_METRICS=1
export RAY_DISABLE_DASHBOARD=1
export RAY_DEDUP_LOGS=1
export NCCL_DEBUG=WARN
```

Reasons:

- `HYDRA_FULL_ERROR=1`: keeps Hydra errors diagnosable.
- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`: avoids protobuf extension
  compatibility issues seen in this Python 3.12 PPU image.
- `RAY_DISABLE_GPU_METRICS=1`: avoids Ray dashboard GPU metric scraping paths
  that assume NVML-style devices.
- `RAY_DISABLE_DASHBOARD=1`: avoids dashboard startup and dashboard-side device
  metric code.
- `RAY_DEDUP_LOGS=1`: reduces repeated Ray worker logs.
- `NCCL_DEBUG=WARN`: suppresses noisy communication INFO logs.

If the verl config supports Ray init kwargs, also set:

```text
+ray_kwargs.ray_init.include_dashboard=False
```

For older verl that does not have this Hydra key, disable the dashboard in the
code path that calls `ray.init`, for example:

```python
ray.init(..., include_dashboard=False)
```

After an abnormal exit:

```bash
ray stop --force || true
```

## Patch 1: Ray Runtime Env In verl

Find where the bundled verl defines Ray runtime environment variables. In newer
verl this is:

```text
verl/trainer/constants_ppo.py
```

In older verl, search:

```bash
grep -RIn "runtime_env\\|TOKENIZERS_PARALLELISM\\|VLLM_LOGGING_LEVEL\\|ray.init" verl
```

Make sure Ray workers inherit these values:

```text
TOKENIZERS_PARALLELISM=true
NCCL_DEBUG=WARN
VLLM_LOGGING_LEVEL=WARN
VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
CUDA_DEVICE_MAX_CONNECTIONS=1
VLLM_DISABLE_COMPILE_CACHE=1
HCCL_HOST_SOCKET_PORT_RANGE=auto
HCCL_NPU_SOCKET_PORT_RANGE=auto
HSA_NO_SCRATCH_RECLAIM=1
```

Important notes:

- `VLLM_DISABLE_COMPILE_CACHE=1` avoids stale or incompatible vLLM compile-cache
  artifacts across runs.
- `VLLM_LOGGING_LEVEL=WARN` keeps vLLM worker logs readable.
- `HCCL_*` and `HSA_NO_SCRATCH_RECLAIM=1` are harmless on the CUDA-compatible
  path and useful for PPU/HCCL-style runtime behavior.

## Patch 2: Device And Ray Placement

Find the device-selection and worker-placement code. Useful searches:

```bash
grep -RIn "torch.cuda.is_available\\|device_name\\|get_accelerator_ids\\|CUDA_VISIBLE_DEVICES\\|ROCR_VISIBLE_DEVICES\\|HIP_VISIBLE_DEVICES" verl
```

Reference paths in newer verl:

```text
verl/utils/device.py
verl/trainer/main_ppo.py
verl/single_controller/base/worker.py
verl/single_controller/ray/base.py
```

Expected behavior on this PPU:

- If `torch.cuda.is_available()` is true, let verl use the CUDA/GPU path.
- Ray placement should request `"GPU"` resources for PPU cards.
- Workers should respect Ray-assigned accelerator ids.
- Avoid manually overriding `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, or
  `ROCR_VISIBLE_DEVICES` in a way that conflicts with Ray placement.

For single-node runs, these must be internally consistent:

```text
trainer.n_gpus_per_node=<cards_used_by_this_job>
actor_rollout_ref.rollout.tensor_model_parallel_size=<tp_size>
```

If the rollout or worker group code chunks batches equally by worker count, the
batch size must be divisible by that worker count.

## Patch 3: vLLM Async Server Settings

Find the vLLM rollout/server integration. Useful searches:

```bash
grep -RIn "AsyncLLM\\|distributed_executor_backend\\|worker_extension_cls\\|enable_chunked_prefill\\|enable_prefix_caching\\|gpu_memory_utilization\\|tensor_parallel_size" verl
```

Reference paths in newer verl:

```text
verl/workers/rollout/vllm_rollout/vllm_async_server.py
verl/workers/rollout/vllm_rollout/vllm_rollout.py
verl/workers/rollout/llm_server.py
```

Settings that have worked on this PPU vLLM stack:

```text
distributed_executor_backend=mp
worker_extension_cls=<verl vLLM worker extension>
dtype=float16
max_num_batched_tokens=8192
enable_chunked_prefill=True
enable_prefix_caching=True
tensor_parallel_size=1 unless explicitly benchmarking TP
```

Notes:

- `distributed_executor_backend=mp` is the stable Ray-managed async rollout
  path in the tested environment.
- Keep `dtype=float16` unless you are deliberately revalidating dtype behavior.
- `tensor_parallel_size=1` preserves vLLM replica count. TP may help tail
  latency for some workloads, but it also reduces replica count.
- `max_model_len` must cover the actual prompt plus response length used by the
  training job.

## Patch 4: vLLM Future `None` Handling

This is a critical code-level fix for rollout weight synchronization.

Find the rollout adapter method that sends control RPCs to the vLLM server. In
newer verl it is similar to:

```text
verl/workers/rollout/vllm_rollout/vllm_rollout.py
```

Search terms:

```bash
grep -RIn "update_weights_from_ipc\\|non_block\\|collective_rpc\\|async_send_weights\\|await future" verl
```

The pattern should be:

```python
async def _execute_method(...):
    if self.rollout_rank != 0:
        return None
    future = self.server_handle.collective_rpc.remote(...)
    return future if non_block else await future
```

During weight update, the returned future must be guarded:

```python
future = await self._execute_method("update_weights_from_ipc", non_block=True, ...)
await sender.async_send_weights(weights)
if future is not None:
    await future
```

Why this matters:

- Only `rollout_rank == 0` talks to the vLLM async server.
- Other rollout ranks legitimately receive `None`.
- Without the guard, non-zero ranks try to `await None`, and the run fails during
  rollout weight synchronization before meaningful training starts.

If the older verl version uses a different weight-sync implementation, apply the
same rule: any branch that does not launch an async remote call must not be
awaited as a future.

## Patch 5: Sequence Length Limits

Long packed sequences can fail during actor forward/backward or log-prob
computation with:

```text
AssertionError: max_token_len must be greater than the sequence length
```

Find the equivalent config keys in the bundled verl. In newer verl these are:

```text
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=<max_tokens>
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=<max_tokens>
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
```

For older verl, search for:

```bash
grep -RIn "ppo_max_token_len_per_gpu\\|log_prob_max_token_len_per_gpu\\|max_token_len\\|use_dynamic_bsz" verl
```

Use Hydra `+key=value` only when the key does not exist in that verl config
schema. Otherwise use `key=value`.

## Expected Warnings

This warning is expected when a vLLM config uses eager mode:

```text
Enforce eager set, disabling torch.compile and CUDAGraphs
Inductor compilation was disabled by user settings
```

It means compile/CUDAGraph optimizations are disabled for that vLLM instance.

This warning is usually non-fatal for dense fp16 transformer rollout:

```text
acext import failed
```

Installing `acext` in the image is preferred, but this warning alone does not
mean verl cannot run.

## Bring-Up Checklist

When adapting a new project-bundled verl to PPU, check in this order:

1. The system Python imports PPU-compatible torch, vLLM, and Ray.
2. `torch.cuda.is_available()` returns the expected value for the image.
3. Runtime env vars are exported before launching verl.
4. `ray.init` disables the dashboard.
5. Ray worker runtime env includes the PPU/vLLM variables above.
6. Device placement uses Ray `"GPU"` resources when torch exposes PPU through
   CUDA-compatible APIs.
7. vLLM rollout uses the async server path with fp16 and `mp` executor backend.
8. Rollout weight sync guards `future is not None` before awaiting.
9. Sequence token limits are large enough for the actual batch.
10. Batch size is divisible by worker/chunk count if that verl path requires
    equal chunks.
