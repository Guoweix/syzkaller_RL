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

type RLSession struct {
    client    *jsonrpc.RPCClient  // 改为指针
    clientMux *sync.Mutex         // 添加锁
    sessionID string
}

type ActionType int

const (
    ActionMutate    ActionType = 0 // 变异
    ActionMerge     ActionType = 1 // 合并
    ActionInsert    ActionType = 2 // 插入
    ActionDelete    ActionType = 3 // 删除
    ActionNormalize ActionType = 4 // 参数规整
)

type Action struct {
    SessionID    string     `json:"session_id"`
    ActionType   ActionType `json:"action_type"`
    ActionParam  int        `json:"action_param"`
}

type ActionState struct {
    SessionID    string `json:"session_id"`
    CallSequence []int  `json:"call_sequence"`
    CallCount    int    `json:"call_count"`
    ExecTime     uint64 `json:"exec_time"`
    ErrorCount   int    `json:"error_count"`
}

func NewRLSession(client *jsonrpc.RPCClient, clientMux *sync.Mutex, prog *prog.Prog) (*RLSession, error) {
    sessionID := hash.String(prog.Serialize())
    
    session := &RLSession{
        client:    client,
        clientMux: clientMux,
        sessionID: sessionID,
    }
    
    if err := session.InitSession(); err != nil {
        return nil, err
    }
    
    return session, nil
}

func (s *RLSession) Ping() error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    s.clientMux.Lock()
    defer s.clientMux.Unlock()
    
    var result map[string]interface{}
    return (*s.client).CallFor(ctx, &result, "ping")
}

func (s *RLSession) InitSession() error {
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()
    
    s.clientMux.Lock()
    defer s.clientMux.Unlock()
    
    var result map[string]interface{}
    return (*s.client).CallFor(ctx, &result, "init_session", map[string]interface{}{
        "session_id": s.sessionID,
    })
}

func (s *RLSession) EndSession() error {
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()
    
    s.clientMux.Lock()
    defer s.clientMux.Unlock()
    
    var result map[string]interface{}
    return (*s.client).CallFor(ctx, &result, "end_session", map[string]interface{}{
        "session_id": s.sessionID,
    })
}

func (s *RLSession) GetAction(state *ActionState) (*Action, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
    defer cancel()
    
    s.clientMux.Lock()
    defer s.clientMux.Unlock()
    
    var result map[string]interface{}
    err := (*s.client).CallFor(ctx, &result, "get_action", map[string]interface{}{
        "session_id": s.sessionID,
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
        SessionID:   s.sessionID,
        ActionType:  ActionType(int(actionType)),
        ActionParam: int(actionParam),
    }, nil
}

func (s *RLSession) SubmitReward(reward float64) error {
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()
    
    s.clientMux.Lock()
    defer s.clientMux.Unlock()
    
    var result map[string]interface{}
    return (*s.client).CallFor(ctx, &result, "submit_reward", map[string]interface{}{
        "session_id": s.sessionID,
        "reward":     reward,
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

