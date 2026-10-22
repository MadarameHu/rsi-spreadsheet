#!/usr/bin/env bash
# The audit payload is read by the host; this task has verification disabled.
set -euo pipefail
printf '%s\n' '{"reward": 0.0, "note": "independent audit uses its complete JSON artifact"}' > "${REWARD_PATH:-/tmp/reward.json}"
