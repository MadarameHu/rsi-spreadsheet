#!/usr/bin/env bash
# Host validates the structured attribution output; verification is disabled.
set -euo pipefail
printf '%s\n' '{"reward": 0.0, "note": "host validates the complete attribution artifact"}' > "${REWARD_PATH:-/tmp/reward.json}"
