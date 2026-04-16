#!/bin/bash

# Define your parts
DIR_PATH="$DRONA_ENV_DIR"
FILE_NAME="configured_check"

# Concatenate to create the whole path
# This handles the case where DIR_PATH doesn't have a trailing slash
FULL_PATH="${DIR_PATH}/configuration/${FILE_NAME}"

# Check if the file exists
if [ -f "$FULL_PATH" ]; then
    echo "CONFIGURED"
else
    echo "NOTCONFIGURED"
fi 
