#!/usr/bin/env bash
#
# Deploy project code, dependencies and data to remote GPU servers (vast.ai compatible).
# No Docker required on the remote — runs Python directly.
#
# Usage:
#   ./scripts/deploy_to_server.sh                    # deploy code + data to all servers
#   ./scripts/deploy_to_server.sh gpu-server-1       # deploy to specific server
#   ./scripts/deploy_to_server.sh --code-only        # only sync code + install deps
#   ./scripts/deploy_to_server.sh --data-only        # only sync data
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/config/servers.yaml"

# Subjects for CI test subset
CI_SUBJECTS=("DB2_s1" "DB2_s12" "DB2_s15" "DB2_s28" "DB2_s39")

# Files/dirs to exclude from code sync
RSYNC_EXCLUDE=(
    "data/DB2_*"
    "experiments_output/"
    "results/"
    "results_collected/"
    "output/"
    "output_model_comparison/"
    "output_model_comparison_improved_000/"
    "research_agent/qdrant_data/"
    "omega_env/"
    "agents_env/"
    ".venv/"
    "__pycache__/"
    ".git/"
    "*.png"
    "*.pdf"
    "*.pyc"
    "docs/"
    "paper_figures/"
)

# Parse config with Python (portable YAML parsing)
parse_servers() {
    python3 -c "
import yaml, json, sys
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
for s in cfg.get('servers', []):
    print(json.dumps(s))
"
}

# --- Parse arguments ---
TARGET_SERVER=""
CODE_ONLY=false
DATA_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --code-only) CODE_ONLY=true; shift ;;
        --data-only) DATA_ONLY=true; shift ;;
        *)           TARGET_SERVER="$1"; shift ;;
    esac
done

# --- Deploy to a single server ---
deploy_to_server() {
    local server_json="$1"
    local name host port user ssh_key work_dir data_dir

    name=$(echo "$server_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['name'])")
    host=$(echo "$server_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['host'])")
    port=$(echo "$server_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('port', 22))")
    user=$(echo "$server_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['user'])")
    ssh_key=$(echo "$server_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh_key', '~/.ssh/id_rsa'))")
    work_dir=$(echo "$server_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['work_dir'])")
    data_dir=$(echo "$server_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data_dir'])")

    # Expand tilde
    ssh_key="${ssh_key/#\~/$HOME}"

    local SSH_CMD="ssh -i $ssh_key -p $port -o StrictHostKeyChecking=no"
    local SSH_TARGET="$user@$host"
    local RSYNC_SSH="ssh -i $ssh_key -p $port -o StrictHostKeyChecking=no"

    echo ""
    echo "=== Deploying to $name ($SSH_TARGET) ==="

    # Create dirs on remote
    $SSH_CMD "$SSH_TARGET" "mkdir -p $work_dir $data_dir $work_dir/experiments_output"

    # --- Sync project code ---
    if [ "$DATA_ONLY" = false ]; then
        echo "  [1/3] Syncing project code..."

        # Build rsync exclude args
        local EXCLUDE_ARGS=""
        for excl in "${RSYNC_EXCLUDE[@]}"; do
            EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$excl"
        done

        rsync -avz --progress --delete \
            -e "$RSYNC_SSH" \
            $EXCLUDE_ARGS \
            "$PROJECT_ROOT/" \
            "$SSH_TARGET:$work_dir/"

        echo "  [2/3] Installing Python dependencies..."
        $SSH_CMD "$SSH_TARGET" "cd $work_dir && pip3 install --break-system-packages --ignore-installed -q -r requirements.txt 2>&1 | tail -5"

        # Create symlinks for each subject: work_dir/data/DB2_sN -> data_dir/DB2_sN
        echo "  [2.5/3] Linking data subjects..."
        $SSH_CMD "$SSH_TARGET" "for d in $data_dir/DB2_*; do ln -sfn \"\$d\" $work_dir/data/\$(basename \"\$d\"); done"

        echo "  Code deployment complete"
    fi

    # --- Sync data (CI test subjects) ---
    if [ "$CODE_ONLY" = false ]; then
        echo "  [3/3] Syncing data (CI test subjects)..."
        for subj in "${CI_SUBJECTS[@]}"; do
            local subj_dir="$PROJECT_ROOT/data/$subj"
            if [ -d "$subj_dir" ]; then
                echo "    Syncing $subj..."
                rsync -avz --progress \
                    -e "$RSYNC_SSH" \
                    "$subj_dir/" \
                    "$SSH_TARGET:$data_dir/$subj/"
            else
                echo "    WARNING: $subj_dir not found locally, skipping"
            fi
        done
        echo "  Data sync complete"
    fi

    echo "  Deploy complete for $name"
}

# --- Iterate servers ---
parse_servers | while read -r server_json; do
    name=$(echo "$server_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['name'])")

    if [ -n "$TARGET_SERVER" ] && [ "$name" != "$TARGET_SERVER" ]; then
        continue
    fi

    deploy_to_server "$server_json"
done

echo ""
echo "=== Deployment complete ==="
