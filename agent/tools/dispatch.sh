#!/bin/sh
# Put one of our engine trees on a merged-team queue: dispatch.sh <mate|silver|dead> <sha> "<message>"
# A merge commit keeps their history (their main is a parent), so the push is a fast-forward.
# Every push starts one official run on THAT team's queue; their result is visible only as
# their leaderboard best.
set -e
case "$1" in
  mate)   url=https://github.com/john-jpet/fast-transformer ;;
  # Silver Bullet is another agent's queue as of 2026-09-20: they are building a
  # new architecture there. The target stays wired up for them; we do not use it.
  silver) url=https://github.com/sivakovivan/silver-transformer ;;
  dead)   url=https://github.com/aparajitamehtatbsw-dot/fast-transform-super-fast ;;
  *) echo "usage: dispatch.sh <mate|silver|dead> <sha> <message>" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/../.."
if git fetch -q "$url" "+main:refs/remotes/$1/main" 2>/dev/null; then
  # Their history is a parent, so the push is a fast-forward and nothing of theirs is rewritten.
  merge=$(git commit-tree "$2^{tree}" -p "$2" -p "refs/remotes/$1/main" -m "$3")
else
  # A repository with no main yet: our tree becomes its first commit.
  merge=$(git commit-tree "$2^{tree}" -p "$2" -m "$3")
fi
git update-ref "refs/heads/dispatch-$1" "$merge"
git push "$url" "dispatch-$1:main"
