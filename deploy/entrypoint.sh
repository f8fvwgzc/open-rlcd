#!/bin/sh
# Container entrypoint: `rlcd "$@"`, after two checks that turn a silent misconfiguration into a start-up error.
set -eu

# 1. RLCD_PROJECT is set but there is no rlcd.yaml there (a missing or mistyped mount). Without this check the
#    service would start as if no project existed and, on an empty home, import and serve the synthetic demo
#    model under the business's hostname.
if [ -n "${RLCD_PROJECT:-}" ]; then
    if [ -d "$RLCD_PROJECT" ]; then cfg="$RLCD_PROJECT/rlcd.yaml"; else cfg="$RLCD_PROJECT"; fi
    if [ ! -f "$cfg" ]; then
        echo "rlcd-entrypoint: RLCD_PROJECT=$RLCD_PROJECT but $cfg does not exist (is the project mounted?)" >&2
        exit 64
    fi
fi

# 2. The home is not writable by this user (a bind mount owned by another uid).
home="${RLCD_HOME:-/var/lib/rlcd}"
if ! { [ -d "$home" ] && [ -w "$home" ]; }; then
    echo "rlcd-entrypoint: RLCD_HOME=$home is not a directory writable by uid $(id -u)" \
         "(for a bind mount: chown -R 10001:10001 it on the host)" >&2
    exit 73
fi

exec rlcd "$@"
