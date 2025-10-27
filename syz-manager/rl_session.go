package main

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/syzkaller/pkg/hash"
	"github.com/google/syzkaller/prog"
	"github.com/ybbus/jsonrpc/v3"
)

type ActionType int

const (
	ActionMutate    ActionType = 0 // 变异
	ActionMerge     ActionType = 1 // 合并
	ActionInsert    ActionType = 2 // 插入
	ActionDelete    ActionType = 3 // 删除
	ActionNormalize ActionType = 4 // 参数规整
	Error           ActionType = 5 // 错误
)

type Action struct {
	SessionID   string     `json:"session_id"`
	ActionType  ActionType `json:"action_type"`
	ActionParam int        `json:"action_param"`
}

type ActionState struct {
	SessionID    string `json:"session_id"`
	CallSequence []int  `json:"call_sequence"`
	CallCount    int    `json:"call_count"`
	ExecTime     uint64 `json:"exec_time"`
	ErrorCount   int    `json:"error_count"`
}

func Ping(client *jsonrpc.RPCClient, clientMux *sync.Mutex) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	clientMux.Lock()
	defer clientMux.Unlock()

	var result map[string]interface{}
	return (*client).CallFor(ctx, &result, "ping")
}

func InitSession(client *jsonrpc.RPCClient, clientMux *sync.Mutex, sessionID string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	clientMux.Lock()
	defer clientMux.Unlock()

	var result map[string]interface{}
	return (*client).CallFor(ctx, &result, "init_session", map[string]interface{}{
		"session_id": sessionID,
	})
}

func GetAction(client *jsonrpc.RPCClient, clientMux *sync.Mutex, sessionID string, state *ActionState) (*Action, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	clientMux.Lock()
	defer clientMux.Unlock()

	var result map[string]interface{}
	err := (*client).CallFor(ctx, &result, "get_action", map[string]interface{}{
		"session_id": sessionID,
		"state":      state,
	})
	if err != nil {
		return nil, err
	}

	// 解析返回的action
	actionData, ok := result["action"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid action format in response")
	}

	actionType, ok := actionData["action_type"].(float64)
	if !ok {
		return nil, fmt.Errorf("invalid action_type in response")
	}

	actionParam, ok := actionData["action_param"].(float64)
	if !ok {
		return nil, fmt.Errorf("invalid action_param in response")
	}

	return &Action{
		SessionID:   sessionID,
		ActionType:  ActionType(int(actionType)),
		ActionParam: int(actionParam),
	}, nil
}

func SubmitReward(client *jsonrpc.RPCClient, clientMux *sync.Mutex, sessionID string, reward float64) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	clientMux.Lock()
	defer clientMux.Unlock()

	var result map[string]interface{}
	return (*client).CallFor(ctx, &result, "submit_reward", map[string]interface{}{
		"session_id": sessionID,
		"reward":     reward,
	})
}

func ChangeSessionID(client *jsonrpc.RPCClient, clientMux *sync.Mutex, oldSessionID string, newSessionID string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	clientMux.Lock()
	defer clientMux.Unlock()
	var result map[string]interface{}
	return (*client).CallFor(ctx, &result, "change_session_id", map[string]interface{}{
		"old_session_id": oldSessionID,
		"new_session_id": newSessionID,
	})
}

func BuildActionState(p *prog.Prog, execTime uint64, errorCount int) *ActionState {
	sig := hash.String(p.Serialize())
	callSequence := make([]int, len(p.Calls))

	for i, call := range p.Calls {
		callSequence[i] = call.Meta.ID
	}

	return &ActionState{
		SessionID:    sig,
		CallSequence: callSequence,
		CallCount:    len(p.Calls),
		ExecTime:     execTime,
		ErrorCount:   errorCount,
	}
}
