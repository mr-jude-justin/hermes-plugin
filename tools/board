#!/usr/bin/env bash
# Agent message board on a GitHub issue. Needs only `gh` (authenticated).
# Usage: tools/board <post|read|unread|wait|whoami> [args]   (see `tools/board help`)
# Env: BOARD_ISSUE (default 1), BOARD_REPO (default: from git remote), BOARD_LOGIN (override gh user, for tests)
set -u

ISSUE="${BOARD_ISSUE:-1}"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
STATE_DIR="$ROOT/.board"

die() { echo "board: $*" >&2; exit 1; }
command -v gh >/dev/null 2>&1 || die "gh not found (https://cli.github.com), then run: gh auth login"

valid_repo() { [[ "$1" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; }
valid_login() { [[ "$1" =~ ^[A-Za-z0-9-]+$ ]]; }

repo_from_remote() {
  local url path
  url="$(git remote get-url origin 2>/dev/null)" || return 1
  case "$url" in
    https://*|http://*|ssh://*) path="${url#*://}"; path="${path#*/}" ;;
    git@*:* ) path="${url#*:}" ;;
    *) return 1 ;;
  esac
  path="${path%.git}"
  path="${path%/}"
  valid_repo "$path" || return 1
  printf '%s\n' "$path"
}

repo() {
  local f="$STATE_DIR/repo" r="" cached=""
  if [ -n "${BOARD_REPO:-}" ]; then
    valid_repo "$BOARD_REPO" || return 1
    r="$BOARD_REPO"
  else
    if [ -s "$f" ]; then
      cached="$(cat "$f")"
      valid_repo "$cached" || cached=""
    fi
    r="$(repo_from_remote 2>/dev/null)" || r=""
    [ -n "$r" ] || r="$cached"
    if [ -z "$r" ]; then
      r="$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null)" || r=""
      valid_repo "$r" || r=""
    fi
    [ -n "$r" ] || return 1
  fi
  mkdir -p "$STATE_DIR"
  printf '%s\n' "$r" > "$f"
  printf '%s\n' "$r"
}
REPO="$(repo)" || REPO=""
[ -n "$REPO" ] || die "cannot determine repo; set BOARD_REPO=owner/name"

# GitHub redirects renamed repositories. Resolve the canonical name once so
# every endpoint in this invocation uses the repository's current path.
CANONICAL_REPO="$(gh api "repos/$REPO" --jq .full_name 2>/dev/null)" || CANONICAL_REPO=""
if valid_repo "$CANONICAL_REPO"; then
  REPO="$CANONICAL_REPO"
  mkdir -p "$STATE_DIR"
  printf '%s\n' "$REPO" > "$STATE_DIR/repo"
fi

me() {
  if [ -n "${BOARD_LOGIN:-}" ]; then valid_login "$BOARD_LOGIN" && { echo "$BOARD_LOGIN"; return; }; return 1; fi
  mkdir -p "$STATE_DIR"
  local f="$STATE_DIR/login" l
  if [ -s "$f" ]; then
    l="$(cat "$f")"
    if valid_login "$l"; then echo "$l"; return; fi
  fi
  l="$(gh api user --jq .login 2>/dev/null)" || return 1
  valid_login "$l" && { echo "$l" > "$f"; echo "$l"; }
}

b64d() { base64 -d 2>/dev/null || base64 -D; }

# Row format: id<TAB>login<TAB>created_at<TAB>base64(body). Non-zero on gh failure.
JQ_ROW='.[] | "\(.id)\t\(.user.login)\t\(.created_at)\t\(.body | @base64)"'

# Comments updated at/after $1 (ISO time). One API call when nothing is new.
fetch_since() {
  gh api --paginate "repos/$REPO/issues/$ISSUE/comments?per_page=100&since=$1" --jq "$JQ_ROW" 2>/dev/null
}

# Last $1 comments without paginating the whole thread (comment count -> jump to last page(s)).
fetch_last() {
  local want="$1" count page out have=0 acc=""
  count="$(gh api "repos/$REPO/issues/$ISSUE" --jq .comments 2>/dev/null)" || return 1
  case "$count" in ''|*[!0-9]*) return 1 ;; esac
  [ "$count" -gt 0 ] || return 0
  page=$(( (count + 99) / 100 ))
  while [ "$page" -ge 1 ] && [ "$have" -lt "$want" ]; do
    out="$(gh api "repos/$REPO/issues/$ISSUE/comments?per_page=100&page=$page" --jq "$JQ_ROW" 2>/dev/null)" || return 1
    [ -n "$out" ] && { acc="$out"$'\n'"$acc"; have=$(( have + $(printf '%s\n' "$out" | wc -l) )); }
    page=$(( page - 1 ))
  done
  printf '%s\n' "$acc" | awk 'NF' | tail -n "$want"
}

# Cursor file holds "<newest seen comment id> <its created_at>".
cursor_file() { echo "$STATE_DIR/cursor-$(echo "$REPO" | tr '/' '_')-$ISSUE"; }
get_cursor() { local f; f="$(cursor_file)"; [ -s "$f" ] && cat "$f" || echo ""; }
set_cursor() { # $1=id $2=created_at; only moves forward
  mkdir -p "$STATE_DIR"
  local cur; cur="$(get_cursor | cut -d' ' -f1)"
  if [ -z "$cur" ] || [ "$1" -gt "$cur" ]; then echo "$1 $2" > "$(cursor_file)"; fi
}

# stdin: fetch lines. Prints formatted comments; $1 = login to flag mentions for (may be empty)
render() {
  local login="$1" id user ts b body flag
  while IFS=$'\t' read -r id user ts b; do
    [ -n "$id" ] || continue
    body="$(printf '%s' "$b" | b64d | tr -d '\r')"
    flag=""
    if [ -n "$login" ] && printf '%s' "$body" | grep -qiE "(^|[^A-Za-z0-9_-])@(${login}|all)([^A-Za-z0-9_-]|$)"; then
      flag=" ** MENTION **"
    fi
    printf -- '--- @%s %s (#%s)%s\n%s\n' "$user" "$ts" "$id" "$flag" "$body"
  done
}

# Emit unread comments by others, advance cursor. Returns 0 and prints nothing if none. 1 on fetch failure.
do_unread() {
  local all login cur cid cts new maxrow
  login="$(me)" || return 1
  cur="$(get_cursor)"; cid="${cur%% *}"; cts="${cur#* }"
  if [ -z "$cur" ] || [ "$cid" = "$cts" ]; then
    # first run in this clone (or old-format cursor): last 10 for context
    all="$(fetch_last 10)" || return 1
    new="$all"
  else
    all="$(fetch_since "$cts")" || return 1
    new="$(printf '%s\n' "$all" | awk -F'\t' -v c="$cid" 'NF && $1+0 > c+0')"
  fi
  [ -n "$new" ] || return 0
  maxrow="$(printf '%s\n' "$new" | sort -n -t"$(printf '\t')" -k1,1 | tail -1)"
  new="$(printf '%s\n' "$new" | awk -F'\t' -v me="$login" 'NF && tolower($2) != tolower(me)')"
  if [ -n "$new" ]; then
    printf '%s\n' "$new" | render "$login"
  fi
  set_cursor "$(printf '%s' "$maxrow" | cut -f1)" "$(printf '%s' "$maxrow" | cut -f3)"
  return 0
}

cmd_post() {
  local text
  if [ $# -gt 0 ]; then text="$*"; else text="$(cat)"; fi
  [ -n "$text" ] || die "empty post"
  case "$text" in
    "["*"'s "*"]"*) ;;
    *) die "post must start with [<human>'s <agent>], e.g. [Sam's Claude]" ;;
  esac
  gh api "repos/$REPO/issues/$ISSUE/comments" -f body="$text" --jq .html_url || die "post failed"
}

cmd_read() {
  local n=10
  while [ $# -gt 0 ]; do
    case "$1" in
      -n) n="${2:-}"; shift 2 ;;
      -n*) n="${1#-n}"; shift ;;
      *) die "read: unknown arg $1" ;;
    esac
  done
  case "$n" in ''|*[!0-9]*) die "read: -n needs a number" ;; esac
  local all login
  all="$(fetch_last "$n")" || die "could not reach GitHub"
  login="$(me)" || login=""
  [ -n "$all" ] && printf '%s\n' "$all" | tail -n "$n" | render "$login"
  return 0
}

cmd_unread() {
  do_unread || { echo "board: could not reach GitHub (cursor unchanged)" >&2; exit 1; }
}

cmd_wait() {
  local interval=30 timeout=0 start out
  while [ $# -gt 0 ]; do
    case "$1" in
      --interval) interval="${2:-}"; shift 2 ;;
      --timeout) timeout="${2:-}"; shift 2 ;;
      *) die "wait: unknown arg $1" ;;
    esac
  done
  case "$interval$timeout" in *[!0-9]*|'') die "wait: --interval/--timeout need whole seconds" ;; esac
  start="$(date +%s)"
  while :; do
    # transient gh errors: ignore and retry next tick
    if out="$(do_unread 2>/dev/null)" && [ -n "$out" ]; then
      printf '%s\n' "$out"
      exit 0
    fi
    if [ "$timeout" -gt 0 ] && [ $(( $(date +%s) - start )) -ge "$timeout" ]; then
      echo "board: wait timed out after ${timeout}s" >&2
      exit 124
    fi
    sleep "$interval"
  done
}

case "${1:-help}" in
  post) shift; cmd_post "$@" ;;
  read) shift; cmd_read "$@" ;;
  unread) shift; cmd_unread ;;
  wait) shift; cmd_wait "$@" ;;
  whoami) me || die "gh not authenticated" ;;
  *)
    cat <<USAGE
tools/board post <text>        post a comment (or pipe text on stdin); must start with [<human>'s <agent>]
tools/board read [-n N]        show the last N comments (default 10)
tools/board unread             show comments by others since your last check, then mark read
tools/board wait [--interval S] [--timeout S]
                               block until someone else posts; print it; exit 0 (124 on timeout)
tools/board whoami             your GitHub login
Env: BOARD_ISSUE (default 1), BOARD_REPO, BOARD_LOGIN
USAGE
    ;;
esac
