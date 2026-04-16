#!/bin/bash
# retrievers/servers_by_region.sh
# Example retriever for testing DynamicSelect

region="${region:-default}"
type="${type:-development}"

# Log for debugging (printed to stderr, not UI)
echo "[INFO] Fetching servers for region=$region and type=$type" >&2

# Return JSON array of {value, label} objects
case "$region" in
  "us-east")
    cat <<EOF
[
  { "value": "srv-east-1", "label": "US-East Server 1 ($type)" },
  { "value": "srv-east-2", "label": "US-East Server 2 ($type)" }
]
EOF
    ;;
  "eu-west")
    cat <<EOF
[
  { "value": "srv-eu-1", "label": "EU-West Server 1 ($type)" },
  { "value": "srv-eu-2", "label": "EU-West Server 2 ($type)" }
]
EOF
    ;;
  *)
    cat <<EOF
[
  { "value": "srv-generic-1", "label": "Generic Server A ($type)" },
  { "value": "srv-generic-2", "label": "Generic Server B ($type)" }
]
EOF
    ;;
esac
