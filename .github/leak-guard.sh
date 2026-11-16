#!/bin/sh
# Fails when a tracked file names a private box, tailnet address, home or scratch path, private tool, personal mail or an AI co-author trailer.
# Each pattern has one character in brackets so that this file does not match itself.
cd "$(git rev-parse --show-toplevel)" || exit 2
bad=0
hits=$(mktemp)
check() {   # check LABEL PCRE
    if git grep -nIP -e "$2" -- . >"$hits"; then
        echo "leak guard: $1"; cat "$hits"; bad=1
    fi
}
check "box name"        '(?i)greengob[l]in|littleb[o]y'
check "tailnet address" '\b100\.(?!64\.1\.)(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]+\.[0-9]+\b'
check "home path"       '/hom[e]/[A-Za-z_][A-Za-z0-9_-]*/'
check "scratch path"    '/scratc[h]/'
check "private tool"    'gc[ ]pack\b|goethe-compile[r]'
check "personal mail"   '(?i)[A-Za-z0-9._%+-]+@(gmail|googlemail|outlook|hotmail|yahoo|proton(mail)?|icloud)\.[a-z]+'
check "AI co-author"    '(?i)co-authored-b[y]:\s*claude'
exit $bad
