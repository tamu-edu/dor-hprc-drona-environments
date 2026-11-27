#!/bin/bash
# Acceleration strategy table: the validator's strategy list (option_generator.sh, same FILE /
# FILTER_ACCEL / FILTER_PKG params) as dynamicTable rows.
cd "$(dirname "$0")" || exit 1
bash ./option_generator.sh | python3 ./strategy_table.py
