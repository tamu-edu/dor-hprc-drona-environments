#!/bin/bash
# find_free_tb_port.sh — prints a free port to stdout for TensorBoard on this node.
# Tries TensorBoard convention range (6006-6050) first, then scans high ports.

TB_PORT=""
for port in $(seq 6006 6050); do
  if ! ss -tuln 2>/dev/null | grep -q ":${port} "; then
    TB_PORT=$port
    break
  fi
done

if [ -z "$TB_PORT" ]; then
  for port in $(seq 6051 65535); do
    if ! ss -tuln 2>/dev/null | grep -q ":${port} "; then
      TB_PORT=$port
      break
    fi
  done
fi

if [ -z "$TB_PORT" ]; then
  echo "ERROR: No free port found on node $(hostname)" >&2
  exit 1
fi

echo "$TB_PORT"
