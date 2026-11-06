#!/usr/bin/env bash
# SessionStart: show unread board posts and tell the agent to keep a background wait running.
# Never fails the session: always exits 0.
cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
out="$(tools/board unread 2>&1)" || out="(board unreachable: $(printf '%s' "$out" | head -n 1)). Run tools/board unread later."
echo "Team board (GitHub issue #${BOARD_ISSUE:-1}). Rules are in AGENTS.md."
if [ -n "$out" ]; then
  echo "Unread posts:"
  echo "$out"
else
  echo "No unread posts."
fi
if [ "${BOARD_WAIT:-}" = "off" ]; then
  echo "Board loop: your supervisor watches the board and relays anything that affects you. Do NOT arm a background tools/board wait. Still run tools/board unread before each commit and after merges."
else
  cat <<'MSG'
Board loop: start ONE background Bash command now with run_in_background: `tools/board wait`. When it completes, its output is new posts: handle them (a MENTION means act and reply on the board; otherwise reassess your own work only if affected, no acknowledgement-only replies), then re-arm by starting `tools/board wait` in the background again. Do not run two waits at once.
MSG
fi
exit 0
