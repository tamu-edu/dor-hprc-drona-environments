#!/bin/bash

JSON_INPUT=$($DRONA_RUNTIME_DIR/db_access/drona_db_retriever.py -i $WORKFLOW_ID)
python3 <<EOF
import json
data = json.loads("""$JSON_INPUT""")
print(data.get("location", ""))
EOF

