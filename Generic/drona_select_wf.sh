#!/bin/bash

export DRONA_ENV=$DRONA_ENV_NAME

export DRONA_ENV=Generic
python3 $DRONA_RUNTIME_DIR/retriever_scripts/drona_select_wf.py

