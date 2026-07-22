#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# RECREATE=1 is intentionally passed through only when the caller explicitly
# sets it. Each child installer otherwise refuses to modify an existing env.
bash "$SCRIPT_DIR/install_browseragent_v2.sh"
bash "$SCRIPT_DIR/install_swift_sft.sh"
