#!/usr/bin/env bash
# WebShop setup for the shared torch290/uv environment.
# This intentionally avoids conda and installs only the dependencies needed by
# the text environment and Pyserini search index.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f /mnt/data/xts/start.sh ]]; then
    # shellcheck disable=SC1091
    source /mnt/data/xts/start.sh
fi

usage() {
    echo "Usage: $0 [-d small|all] [--skip-deps] [--skip-data] [--skip-index] [--skip-spacy]"
    exit 1
}

data=""
skip_data=false
skip_deps=false
skip_index=false
skip_spacy=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d)
            data="${2:-}"
            shift 2
            ;;
        --skip-data)
            skip_data=true
            shift
            ;;
        --skip-deps)
            skip_deps=true
            shift
            ;;
        --skip-index)
            skip_index=true
            shift
            ;;
        --skip-spacy)
            skip_spacy=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            usage
            ;;
    esac
done

if [[ "$skip_data" != true && -z "$data" ]]; then
    echo "[ERROR] Missing -d flag"
    usage
fi

if [[ -f /mnt/data/xts/setup_env.sh ]]; then
    # shellcheck disable=SC1091
    source /mnt/data/xts/setup_env.sh
    activate_torch290
fi

if [[ "$skip_deps" != true ]]; then
    python -m pip install \
        beautifulsoup4==4.11.1 \
        cleantext==1.1.4 \
        Cython \
        Flask==2.1.2 \
        gdown \
        gym==0.24.0 \
        onnxruntime \
        pyjnius \
        PyYAML==6.0.2 \
        python-Levenshtein \
        rank_bm25==0.2.2 \
        requests_mock \
        "rich>=13.8.0" \
        scikit_learn \
        selenium==4.2.0 \
        thefuzz==0.19.0 \
        "tqdm>=4.66.3" \
        Werkzeug==2.1.2

    python -m pip install --no-deps pyserini==0.17.0

    if ! command -v javac >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update
            apt-get install -y openjdk-17-jdk-headless
        else
            echo "[ERROR] JDK is required by Pyserini, but javac and apt-get are unavailable."
            exit 1
        fi
    fi
fi

if [[ "$skip_data" != true ]]; then
    mkdir -p data
    pushd data >/dev/null

    fetch() {
        local fname="$1"
        local gid="$2"
        if [[ -f "$fname" ]]; then
            echo "[setup_no_conda] $fname already present, skipping download"
        else
            gdown "https://drive.google.com/uc?id=${gid}" -O "$fname"
        fi
    }

    if [[ "$data" == "small" ]]; then
        fetch items_shuffle_1000.json        1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib
        fetch items_ins_v2_1000.json         1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu
    elif [[ "$data" == "all" ]]; then
        fetch items_shuffle_1000.json        1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib
        fetch items_ins_v2_1000.json         1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu
        fetch items_shuffle.json             1A2whVgOO0euk5O13n2iYDM0bQRkkRduB
        fetch items_ins_v2.json              1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi
    else
        echo "[ERROR] argument for -d must be small or all"
        usage
    fi
    fetch items_human_ins.json               14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O
    popd >/dev/null
fi

if [[ "$skip_spacy" != true ]]; then
    python -m spacy download en_core_web_sm
fi

if [[ "$skip_index" != true ]]; then
    pushd search_engine >/dev/null
    mkdir -p resources resources_100 resources_1k resources_100k indexes
    python convert_product_file_format.py
    ./run_indexing.sh
    popd >/dev/null
fi
