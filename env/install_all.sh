#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/install_browseragent_v2.sh"
# bash "$SCRIPT_DIR/install_vllm_server.sh"
bash "$SCRIPT_DIR/install_swift_sft.sh"
