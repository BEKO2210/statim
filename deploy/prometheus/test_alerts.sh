#!/bin/sh
set -eu

PROMTOOL=${PROMTOOL:-promtool}
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "$PROMTOOL" in
    /*) ;;
    */*) PROMTOOL=$(CDPATH= cd -- "$(dirname -- "$PROMTOOL")" && pwd)/$(basename -- "$PROMTOOL") ;;
esac

"$PROMTOOL" check rules "$SCRIPT_DIR/statim-alerts.yml"
(
    cd "$SCRIPT_DIR"
    "$PROMTOOL" test rules statim-alerts.test.yml
)
