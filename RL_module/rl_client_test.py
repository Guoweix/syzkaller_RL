#!/usr/bin/env python3
"""
JSON-RPC Client for testing RL Service
"""

import requests
import json


def json_rpc_request(url, method, params=None, request_id=1):
    """Make a JSON-RPC request"""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "id": request_id
    }
    if params:
        payload["params"] = params
    
    try:
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        return response.json()
    except Exception as e:
        return {"error": f"Request failed: {e}"}


def run_client():
    """Test the JSON-RPC RL service"""
    server_url = "http://localhost:5000"
    
    print("Testing JSON-RPC RL Service...")
    
    # Test Ping
    print("\n1. Testing Ping:")
    response = json_rpc_request(server_url, "ping")
    if "result" in response:
        result = response["result"]
        print(f"Status: {result.get('status')}, Timestamp: {result.get('timestamp')}")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    
    # Test InitSession
    print("\n2. Testing InitSession:")
    response = json_rpc_request(server_url, "init_session", {"session_id": "test_session"})
    if "result" in response:
        result = response["result"]
        print(f"Success: {result.get('success')}, Message: {result.get('message')}")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    
    # Test EndSession
    print("\n3. Testing EndSession:")
    response = json_rpc_request(server_url, "end_session", {"session_id": "test_session"})
    if "result" in response:
        result = response["result"]
        print(f"Success: {result.get('success')}, Message: {result.get('message')}")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    
    # Test InitSession again
    print("\n4. Testing InitSession (again):")
    response = json_rpc_request(server_url, "init_session", {"session_id": "test_session2"})
    if "result" in response:
        result = response["result"]
        print(f"Success: {result.get('success')}, Message: {result.get('message')}")
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    
    # Test EndSession for non-existent session
    print("\n5. Testing EndSession (non-existent):")
    response = json_rpc_request(server_url, "end_session", {"session_id": "non_existent"})
    if "result" in response:
        result = response["result"]
        print(f"Success: {result.get('success')}, Message: {result.get('message')}")
    else:
        error = response.get('error', {})
        print(f"Error Code: {error.get('code')}, Message: {error.get('message')}")


if __name__ == '__main__':
    run_client()