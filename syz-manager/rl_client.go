// Copyright 2025 syzkaller project authors. All rights reserved.
// Use of this source code is governed by Apache 2 LICENSE that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"time"

	"github.com/google/syzkaller/pkg/log"
)

// RLClient represents a client connection to the RL RPC server
type RLClient struct {
	address string
	timeout time.Duration
}

// RLRequest represents a JSON-RPC request
type RLRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
	ID      int         `json:"id"`
}

// RLResponse represents a JSON-RPC response
type RLResponse struct {
	JSONRPC string      `json:"jsonrpc"`
	Result  interface{} `json:"result,omitempty"`
	Error   *RLError    `json:"error,omitempty"`
	ID      int         `json:"id"`
}

// RLError represents a JSON-RPC error
type RLError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// NewRLClient creates a new RL RPC client
func NewRLClient(address string) *RLClient {
	return &RLClient{
		address: address,
		timeout: 10 * time.Second,
	}
}

// SetTimeout sets the timeout for RPC calls
func (c *RLClient) SetTimeout(timeout time.Duration) {
	c.timeout = timeout
}

// Call makes an RPC call to the RL server
func (c *RLClient) Call(method string, params interface{}) (*RLResponse, error) {
	// Create request
	request := &RLRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
		ID:      1, // Simple static ID for now
	}

	// Marshal request
	requestData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	// Connect to server
	conn, err := net.DialTimeout("tcp", c.address, c.timeout)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to RL server at %s: %v", c.address, err)
	}
	defer conn.Close()

	// Set read/write deadlines
	conn.SetDeadline(time.Now().Add(c.timeout))

	// Send request
	_, err = conn.Write(requestData)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %v", err)
	}

	// Read response
	buffer := make([]byte, 4096)
	n, err := conn.Read(buffer)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %v", err)
	}

	// Trim response (remove newlines)
	responseData := bytes.TrimSpace(buffer[:n])

	// Unmarshal response
	var response RLResponse
	err = json.Unmarshal(responseData, &response)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %v", err)
	}

	return &response, nil
}

// HelloWorld calls the hello_world method on the RL server
func (c *RLClient) HelloWorld(name string) (map[string]interface{}, error) {
	params := map[string]interface{}{
		"name": name,
	}

	response, err := c.Call("hello_world", params)
	if err != nil {
		return nil, err
	}

	if response.Error != nil {
		return nil, fmt.Errorf("RPC error %d: %s", response.Error.Code, response.Error.Message)
	}

	result, ok := response.Result.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected response type")
	}

	return result, nil
}

// Ping calls the ping method on the RL server
func (c *RLClient) Ping() (map[string]interface{}, error) {
	response, err := c.Call("ping", nil)
	if err != nil {
		return nil, err
	}

	if response.Error != nil {
		return nil, fmt.Errorf("RPC error %d: %s", response.Error.Code, response.Error.Message)
	}

	result, ok := response.Result.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected response type")
	}

	return result, nil
}

// GetServerInfo calls the get_server_info method on the RL server
func (c *RLClient) GetServerInfo() (map[string]interface{}, error) {
	response, err := c.Call("get_server_info", nil)
	if err != nil {
		return nil, err
	}

	if response.Error != nil {
		return nil, fmt.Errorf("RPC error %d: %s", response.Error.Code, response.Error.Message)
	}

	result, ok := response.Result.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected response type")
	}

	return result, nil
}

// IsAvailable checks if the RL server is available
func (c *RLClient) IsAvailable() bool {
	_, err := c.Ping()
	return err == nil
}

// RLManager handles RL integration for the manager
type RLManager struct {
	client  *RLClient
	enabled bool
}

// NewRLManager creates a new RL manager
func NewRLManager(rlServerAddr string) *RLManager {
	if rlServerAddr == "" {
		return &RLManager{enabled: false}
	}

	client := NewRLClient(rlServerAddr)
	client.SetTimeout(5 * time.Second) // Shorter timeout for manager

	rlMgr := &RLManager{
		client:  client,
		enabled: true,
	}

	// Test connection
	if !client.IsAvailable() {
		log.Logf(1, "RL server not available at %s, disabling RL integration", rlServerAddr)
		rlMgr.enabled = false
	} else {
		log.Logf(0, "RL server connected at %s", rlServerAddr)
		// Get server info
		if info, err := client.GetServerInfo(); err == nil {
			log.Logf(1, "RL server info: %+v", info)
		}
	}

	return rlMgr
}

// TestConnection tests the connection to the RL server
func (rm *RLManager) TestConnection() error {
	if !rm.enabled {
		return fmt.Errorf("RL integration is disabled")
	}

	// Test hello world
	result, err := rm.client.HelloWorld("syz-manager")
	if err != nil {
		return fmt.Errorf("hello world test failed: %v", err)
	}

	log.Logf(1, "RL connection test successful: %s", result["message"])
	return nil
}

// IsEnabled returns whether RL integration is enabled
func (rm *RLManager) IsEnabled() bool {
	return rm.enabled
}

// GetClient returns the RPC client (for future extensions)
func (rm *RLManager) GetClient() *RLClient {
	if !rm.enabled {
		return nil
	}
	return rm.client
}