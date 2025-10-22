#!/bin/bash
# Start RL RPC Server for Syzkaller

echo "Starting RL RPC Server..."
cd "$(dirname "$0")"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed"
    exit 1
fi

# Start the server
python3 rpc_server.py --host 127.0.0.1 --port 9999 "$@"