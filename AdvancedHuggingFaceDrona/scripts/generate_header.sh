#!/bin/bash

# Check if argument is provided
if [ -z "$HEADER" ]; then
    echo "Usage: $0 with HEADER env variable = 'Header Text'"
    exit 1
fi

# Generate the header value string
cat << EOF
<div style='background: #f8f9fa; border: 1px solid #dee2e6; padding: 12px; border-radius: 6px; margin-bottom: 15px; width: 100%;'><h4 style='color: #495057; margin: 0; font-size: 1.1em;'>$HEADER</h4></div>
EOF
exit 0
