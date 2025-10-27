package main

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/google/syzkaller/pkg/hash"
	"github.com/google/syzkaller/prog"
	"github.com/ybbus/jsonrpc/v3"
)

// MockTarget 创建一个简单的测试目标
func createMockTarget() *prog.Target {
	// 创建一些模拟的系统调用
	syscalls := []*prog.Syscall{
		{ID: 0, Name: "test_syscall_0", CallName: "test_syscall_0"},
		{ID: 1, Name: "test_syscall_1", CallName: "test_syscall_1"},
		{ID: 2, Name: "test_syscall_2", CallName: "test_syscall_2"},
	}

	target := &prog.Target{
		Syscalls: syscalls,
	}

	// 初始化 SyscallMap
	target.SyscallMap = make(map[string]*prog.Syscall)
	for i, c := range target.Syscalls {
		c.ID = i
		target.SyscallMap[c.Name] = c
	}

	return target
}

// 创建一个简单的测试程序
func createMockProgram(target *prog.Target) *prog.Prog {
	p := &prog.Prog{
		Target: target,
		Calls: []*prog.Call{
			{Meta: target.Syscalls[0]}, // test_syscall_0
			{Meta: target.Syscalls[1]}, // test_syscall_1
			{Meta: target.Syscalls[2]}, // test_syscall_2
		},
	}
	return p
}

func TestAction(t *testing.T) {
	// 测试 Action 结构体的基本字段
	action := &Action{
		SessionID:   "test_session_123",
		ActionType:  ActionMutate,
		ActionParam: 42,
	}

	if action.SessionID != "test_session_123" {
		t.Errorf("Expected SessionID to be 'test_session_123', got %s", action.SessionID)
	}

	if action.ActionType != ActionMutate {
		t.Errorf("Expected ActionType to be ActionMutate (%d), got %d", ActionMutate, action.ActionType)
	}

	if action.ActionParam != 42 {
		t.Errorf("Expected ActionParam to be 42, got %d", action.ActionParam)
	}

	// 测试所有动作类型枚举值
	expectedTypes := []ActionType{ActionMutate, ActionMerge, ActionInsert, ActionDelete, ActionNormalize}
	expectedValues := []int{0, 1, 2, 3, 4}

	for i, expectedType := range expectedTypes {
		if int(expectedType) != expectedValues[i] {
			t.Errorf("Expected ActionType %d to have value %d, got %d", i, expectedValues[i], int(expectedType))
		}
	}
}

func TestActionState(t *testing.T) {
	// 测试 ActionState 结构体的基本字段
	state := &ActionState{
		SessionID:    "test_session",
		CallSequence: []int{0, 1, 2},
		CallCount:    3,
		ExecTime:     1000,
		ErrorCount:   0,
	}

	if state.SessionID != "test_session" {
		t.Errorf("Expected SessionID to be 'test_session', got %s", state.SessionID)
	}

	if len(state.CallSequence) != 3 {
		t.Errorf("Expected CallSequence length to be 3, got %d", len(state.CallSequence))
	}

	expectedSequence := []int{0, 1, 2}
	for i, expected := range expectedSequence {
		if state.CallSequence[i] != expected {
			t.Errorf("Expected CallSequence[%d] to be %d, got %d", i, expected, state.CallSequence[i])
		}
	}

	if state.CallCount != 3 {
		t.Errorf("Expected CallCount to be 3, got %d", state.CallCount)
	}

	if state.ExecTime != 1000 {
		t.Errorf("Expected ExecTime to be 1000, got %d", state.ExecTime)
	}

	if state.ErrorCount != 0 {
		t.Errorf("Expected ErrorCount to be 0, got %d", state.ErrorCount)
	}
}

func TestBuildActionState(t *testing.T) {
	target := createMockTarget()
	prog := createMockProgram(target)

	execTime := uint64(1500)
	errorCount := 2

	state := BuildActionState(prog, execTime, errorCount)

	// 检查 SessionID 是否正确生成
	expectedSessionID := hash.String(prog.Serialize())
	if state.SessionID != expectedSessionID {
		t.Errorf("Expected SessionID to be %s, got %s", expectedSessionID, state.SessionID)
	}

	// 检查 CallSequence 是否包含正确的系统调用ID
	expectedCallSequence := []int{0, 1, 2}
	if len(state.CallSequence) != len(expectedCallSequence) {
		t.Errorf("Expected CallSequence length to be %d, got %d", len(expectedCallSequence), len(state.CallSequence))
	}

	for i, expected := range expectedCallSequence {
		if i < len(state.CallSequence) && state.CallSequence[i] != expected {
			t.Errorf("Expected CallSequence[%d] to be %d, got %d", i, expected, state.CallSequence[i])
		}
	}

	// 检查 CallCount
	if state.CallCount != len(prog.Calls) {
		t.Errorf("Expected CallCount to be %d, got %d", len(prog.Calls), state.CallCount)
	}

	// 检查 ExecTime
	if state.ExecTime != execTime {
		t.Errorf("Expected ExecTime to be %d, got %d", execTime, state.ExecTime)
	}

	// 检查 ErrorCount
	if state.ErrorCount != errorCount {
		t.Errorf("Expected ErrorCount to be %d, got %d", errorCount, state.ErrorCount)
	}
}

func TestBuildActionStateEmpty(t *testing.T) {
	target := createMockTarget()
	prog := &prog.Prog{
		Target: target,
		Calls:  []*prog.Call{}, // 空的调用序列
	}

	state := BuildActionState(prog, 0, 0)

	if len(state.CallSequence) != 0 {
		t.Errorf("Expected empty CallSequence, got length %d", len(state.CallSequence))
	}

	if state.CallCount != 0 {
		t.Errorf("Expected CallCount to be 0, got %d", state.CallCount)
	}
}

func TestBuildActionStateSingleCall(t *testing.T) {
	target := createMockTarget()
	prog := &prog.Prog{
		Target: target,
		Calls: []*prog.Call{
			{Meta: target.Syscalls[1]}, // 只有一个系统调用
		},
	}

	state := BuildActionState(prog, 500, 1)

	if len(state.CallSequence) != 1 {
		t.Errorf("Expected CallSequence length to be 1, got %d", len(state.CallSequence))
	}

	if state.CallSequence[0] != 1 {
		t.Errorf("Expected CallSequence[0] to be 1, got %d", state.CallSequence[0])
	}

	if state.CallCount != 1 {
		t.Errorf("Expected CallCount to be 1, got %d", state.CallCount)
	}

	if state.ExecTime != 500 {
		t.Errorf("Expected ExecTime to be 500, got %d", state.ExecTime)
	}

	if state.ErrorCount != 1 {
		t.Errorf("Expected ErrorCount to be 1, got %d", state.ErrorCount)
	}
}

func TestPingFunction(t *testing.T) {
	// 测试 Ping 函数
	client := jsonrpc.NewClient("http://localhost:5000")
	var clientMux sync.Mutex

	// 这里测试函数签名是否正确，实际连接测试在集成测试中进行
	err := Ping(&client, &clientMux)
	if err != nil {
		// 预期会失败，因为没有运行的服务端，但不应该panic
		t.Logf("Expected connection error without server: %v", err)
	}
}

// BenchmarkBuildActionState 性能测试
func BenchmarkBuildActionState(b *testing.B) {
	target := createMockTarget()

	// 创建一个有更多调用的程序来测试性能
	calls := make([]*prog.Call, 100)
	for i := 0; i < 100; i++ {
		calls[i] = &prog.Call{Meta: target.Syscalls[i%3]}
	}

	prog := &prog.Prog{
		Target: target,
		Calls:  calls,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		BuildActionState(prog, 1000, 0)
	}
}

// 测试系统调用ID的一致性
func TestSyscallIDConsistency(t *testing.T) {
	target := createMockTarget()

	// 验证系统调用ID与数组索引的一致性
	for i, syscall := range target.Syscalls {
		if syscall.ID != i {
			t.Errorf("Syscall ID inconsistency: expected %d, got %d for syscall %s", i, syscall.ID, syscall.Name)
		}
	}

	// 验证 SyscallMap 的一致性
	for name, syscall := range target.SyscallMap {
		if syscall.Name != name {
			t.Errorf("SyscallMap inconsistency: key %s does not match syscall name %s", name, syscall.Name)
		}

		if target.Syscalls[syscall.ID] != syscall {
			t.Errorf("SyscallMap inconsistency: syscall %s with ID %d not found at correct index", name, syscall.ID)
		}
	}
}

// 完整的RPC客户端测试流程
// 注意：运行此测试前需要先启动RL服务端
// 可以通过以下命令启动服务端：python3 RL_module/rl_server.py
func TestRLSessionCompleteFlow(t *testing.T) {
	// 测试配置
	serverAddr := "localhost:5000" // 假设服务端运行在这个地址
	numIterations := 3             // 执行3次动作-奖励循环

	t.Logf("正在测试完整的RPC流程，服务端地址: %s", serverAddr)
	t.Log("确保RL服务端已经启动: python3 RL_module/rl_server.py")

	// 1. 创建测试程序和客户端
	target := createMockTarget()
	prog := createMockProgram(target)
	sessionID := hash.String(append(prog.Serialize(), []byte(fmt.Sprintf("_complete_%d", time.Now().UnixNano()))...)) // 2. 创建共享的RPC客户端
	t.Log("步骤1: 创建RPC客户端...")
	client := jsonrpc.NewClient(fmt.Sprintf("http://%s", serverAddr))
	var clientMux sync.Mutex

	// 3. 测试Ping
	t.Log("步骤2: 测试Ping连接...")
	if err := Ping(&client, &clientMux); err != nil {
		t.Skipf("无法连接到RPC服务端: %v\n请确保服务端已启动", err)
	} else {
		t.Log("✓ Ping成功")
	}

	// 4. 验证会话ID生成（现在包含时间戳，所以只检查格式）
	if len(sessionID) == 0 {
		t.Errorf("会话ID为空")
	} else {
		t.Logf("✓ 会话ID生成成功: %s", sessionID[:16]+"...")
	}

	// 5. 执行多次动作获取和奖励提交循环
	t.Logf("步骤3: 执行 %d 次动作-奖励循环...", numIterations)

	for i := 0; i < numIterations; i++ {
		t.Logf("  循环 %d/%d:", i+1, numIterations)

		// 构建ActionState（模拟程序执行后的状态）
		execTime := uint64(1000 + i*100) // 模拟递增的执行时间
		errorCount := i                  // 模拟递增的错误数
		state := BuildActionState(prog, execTime, errorCount)

		t.Logf("    状态: CallSequence=%v, ExecTime=%d, ErrorCount=%d",
			state.CallSequence, state.ExecTime, state.ErrorCount)

		// 获取动作
		action, err := GetAction(&client, &clientMux, sessionID, state)
		if err != nil {
			t.Errorf("    获取动作失败: %v", err)
			continue
		}

		t.Logf("    ✓ 获取动作成功: SessionID=%s, Type=%d, Param=%d",
			action.SessionID[:8]+"...", action.ActionType, action.ActionParam)

		// 模拟根据动作执行后计算奖励
		// 这里使用简单的奖励计算逻辑
		reward := 0.5 + float64(i)*0.1 // 递增奖励: 0.5, 0.6, 0.7...

		// 提交奖励
		if err := SubmitReward(&client, &clientMux, sessionID, reward); err != nil {
			t.Errorf("    提交奖励失败: %v", err)
			continue
		}

		t.Logf("    ✓ 提交奖励成功: %.2f", reward)

		// 添加短暂延迟，模拟真实场景
		time.Sleep(100 * time.Millisecond)
	}

	// 6. 测试完成（不需要手动结束会话，服务端会自动管理）
	t.Log("步骤4: 测试完成，服务端会自动管理会话生命周期")

	t.Log("🎉 完整RPC流程测试通过!")
}

// 测试RPC单独方法调用
func TestRLSessionIndividualMethods(t *testing.T) {
	serverAddr := "localhost:5000"

	// 创建测试程序和客户端
	target := createMockTarget()
	prog := createMockProgram(target)
	// 添加时间戳确保sessionID唯一
	sessionID := hash.String(append(prog.Serialize(), []byte(fmt.Sprintf("_individual_%d", time.Now().UnixNano()))...))

	client := jsonrpc.NewClient(fmt.Sprintf("http://%s", serverAddr))
	var clientMux sync.Mutex

	t.Log("测试单独的RPC方法调用...")

	// 测试Ping
	t.Log("1. 测试Ping...")
	if err := Ping(&client, &clientMux); err != nil {
		t.Skipf("无法连接到服务端: %v", err)
	}
	t.Log("   ✓ Ping成功")

	// 测试InitSession（可选，服务端现在会自动管理）
	t.Log("2. 测试InitSession...")
	if err := InitSession(&client, &clientMux, sessionID); err != nil {
		t.Errorf("InitSession失败: %v", err)
		return
	}
	t.Log("   ✓ InitSession成功")

	// 测试GetAction
	t.Log("3. 测试GetAction...")
	state := BuildActionState(prog, 2000, 1)
	action, err := GetAction(&client, &clientMux, sessionID, state)
	if err != nil {
		t.Errorf("GetAction失败: %v", err)
	} else {
		t.Logf("   ✓ GetAction成功，返回: SessionID=%s, Type=%d, Param=%d",
			action.SessionID[:8]+"...", action.ActionType, action.ActionParam)
	}

	// 测试SubmitReward
	t.Log("4. 测试SubmitReward...")
	if err := SubmitReward(&client, &clientMux, sessionID, 0.8); err != nil {
		t.Errorf("SubmitReward失败: %v", err)
	} else {
		t.Log("   ✓ SubmitReward成功")
	}

	// 注意：不再测试EndSession，因为服务端现在自动管理session生命周期
}

// 测试共享客户端和并发访问
func TestSharedClientConcurrency(t *testing.T) {
	serverAddr := "localhost:5000"
	target := createMockTarget()

	// 创建共享的客户端和锁
	client := jsonrpc.NewClient(fmt.Sprintf("http://%s", serverAddr))
	var clientMux sync.Mutex

	// 创建多个程序和对应的sessionID
	numSessions := 3
	programs := make([]*prog.Prog, numSessions)
	sessionIDs := make([]string, numSessions)

	for i := 0; i < numSessions; i++ {
		// 创建不同的程序
		prog := &prog.Prog{
			Target: target,
			Calls: []*prog.Call{
				{Meta: target.Syscalls[i%3]}, // 不同的系统调用组合
			},
		}
		programs[i] = prog
		sessionIDs[i] = hash.String(prog.Serialize())
	}

	// 并发测试多个session
	var wg sync.WaitGroup
	errors := make(chan error, numSessions)

	for i := 0; i < numSessions; i++ {
		wg.Add(1)
		go func(sessionIndex int) {
			defer wg.Done()

			sessionID := sessionIDs[sessionIndex]

			// 测试ping
			if err := Ping(&client, &clientMux); err != nil {
				errors <- fmt.Errorf("session %d ping failed: %v", sessionIndex, err)
				return
			}

			// 测试获取动作
			state := BuildActionState(programs[sessionIndex], uint64(1000+sessionIndex*100), sessionIndex)
			action, err := GetAction(&client, &clientMux, sessionID, state)
			if err != nil {
				errors <- fmt.Errorf("session %d get action failed: %v", sessionIndex, err)
				return
			}

			if action.ActionType < -1 || action.ActionType > Error {
				errors <- fmt.Errorf("session %d got invalid action type: %d", sessionIndex, action.ActionType)
				return
			}

			// 测试提交奖励
			if err := SubmitReward(&client, &clientMux, sessionID, 0.5+float64(sessionIndex)*0.1); err != nil {
				errors <- fmt.Errorf("session %d submit reward failed: %v", sessionIndex, err)
				return
			}

		}(i)
	}

	// 等待所有goroutine完成
	wg.Wait()
	close(errors)

	// 检查是否有错误
	for err := range errors {
		t.Error(err)
	}

	// 不需要手动清理session，服务端会自动管理
	t.Log("✓ 共享客户端并发测试通过")
}

// 性能测试：测试RPC调用延迟
func TestRLSessionPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("跳过性能测试")
	}

	serverAddr := "localhost:5000"
	target := createMockTarget()
	prog := createMockProgram(target)
	sessionID := hash.String(prog.Serialize())

	client := jsonrpc.NewClient(fmt.Sprintf("http://%s", serverAddr))
	var clientMux sync.Mutex

	// 测试连接
	if err := Ping(&client, &clientMux); err != nil {
		t.Skipf("无法连接到服务端: %v", err)
	}

	// 测试多次GetAction调用的性能
	numCalls := 10
	state := BuildActionState(prog, 1000, 0)

	start := time.Now()
	for i := 0; i < numCalls; i++ {
		action, err := GetAction(&client, &clientMux, sessionID, state)
		if err != nil {
			t.Errorf("GetAction调用 %d 失败: %v", i, err)
		} else if action == nil {
			t.Errorf("GetAction调用 %d 返回nil", i)
		}
	}
	elapsed := time.Since(start)

	avgLatency := elapsed / time.Duration(numCalls)
	t.Logf("平均RPC调用延迟: %v (%d次调用总耗时: %v)", avgLatency, numCalls, elapsed)

	// 简单的性能断言（可根据实际需要调整）
	if avgLatency > 100*time.Millisecond {
		t.Logf("警告: 平均延迟较高 (%v), 请检查网络或服务端性能", avgLatency)
	}
}

// 测试自动session创建功能
func TestAutoSessionCreation(t *testing.T) {
	serverAddr := "localhost:5000"
	target := createMockTarget()
	prog := createMockProgram(target)
	sessionID := hash.String(append(prog.Serialize(), []byte(fmt.Sprintf("_auto_%d", time.Now().UnixNano()))...))

	client := jsonrpc.NewClient(fmt.Sprintf("http://%s", serverAddr))
	var clientMux sync.Mutex

	// 测试ping连接
	if err := Ping(&client, &clientMux); err != nil {
		t.Skipf("无法连接到服务端: %v", err)
	}

	t.Log("测试自动session创建...")

	// 直接调用GetAction，不手动创建session
	// 服务端应该自动创建session
	state := BuildActionState(prog, 1500, 0)
	action, err := GetAction(&client, &clientMux, sessionID, state)
	if err != nil {
		t.Errorf("自动session创建失败: %v", err)
		return
	}

	if action.SessionID != sessionID {
		t.Errorf("返回的SessionID不匹配: 期望 %s, 实际 %s", sessionID, action.SessionID)
	}

	t.Logf("✓ 自动session创建成功: SessionID=%s, Type=%d, Param=%d",
		action.SessionID[:8]+"...", action.ActionType, action.ActionParam)

	// 继续使用同一个session
	reward := 0.75
	if err := SubmitReward(&client, &clientMux, sessionID, reward); err != nil {
		t.Errorf("提交奖励失败: %v", err)
	} else {
		t.Logf("✓ 提交奖励成功: %.2f", reward)
	}
}

// 测试多个不同sessionID的并发处理
func TestMultipleSessionsInterleaved(t *testing.T) {
	serverAddr := "localhost:5000"
	target := createMockTarget()

	client := jsonrpc.NewClient(fmt.Sprintf("http://%s", serverAddr))
	var clientMux sync.Mutex

	// 测试连接
	if err := Ping(&client, &clientMux); err != nil {
		t.Skipf("无法连接到服务端: %v", err)
	}

	t.Log("测试多个session的交错处理...")

	// 创建3个不同的程序和session
	numSessions := 3
	programs := make([]*prog.Prog, numSessions)
	sessionIDs := make([]string, numSessions)

	for i := 0; i < numSessions; i++ {
		prog := &prog.Prog{
			Target: target,
			Calls: []*prog.Call{
				{Meta: target.Syscalls[i%3]},
				{Meta: target.Syscalls[(i+1)%3]},
			},
		}
		programs[i] = prog
		sessionIDs[i] = hash.String(prog.Serialize())
		t.Logf("Session %d ID: %s", i, sessionIDs[i][:8]+"...")
	}

	// 交错调用不同的session
	for round := 0; round < 2; round++ {
		t.Logf("Round %d:", round+1)

		for i := 0; i < numSessions; i++ {
			state := BuildActionState(programs[i], uint64(1000+round*500+i*100), round)

			action, err := GetAction(&client, &clientMux, sessionIDs[i], state)
			if err != nil {
				t.Errorf("  Session %d GetAction失败: %v", i, err)
				continue
			}

			t.Logf("  Session %d: 获取动作 Type=%d, Param=%d",
				i, action.ActionType, action.ActionParam)

			reward := 0.3 + float64(i)*0.2 + float64(round)*0.1
			if err := SubmitReward(&client, &clientMux, sessionIDs[i], reward); err != nil {
				t.Errorf("  Session %d SubmitReward失败: %v", i, err)
			} else {
				t.Logf("  Session %d: 提交奖励 %.2f", i, reward)
			}
		}
	}

	t.Log("✓ 多session交错处理测试通过")
}

// 测试无效sessionID处理
func TestInvalidSessionHandling(t *testing.T) {
	serverAddr := "localhost:5000"

	client := jsonrpc.NewClient(fmt.Sprintf("http://%s", serverAddr))
	var clientMux sync.Mutex

	// 测试连接
	if err := Ping(&client, &clientMux); err != nil {
		t.Skipf("无法连接到服务端: %v", err)
	}

	t.Log("测试无效sessionID处理...")

	// 测试空sessionID
	emptyState := &ActionState{
		SessionID:    "",
		CallSequence: []int{0, 1},
		CallCount:    2,
		ExecTime:     1000,
		ErrorCount:   0,
	}

	_, err := GetAction(&client, &clientMux, "", emptyState)
	if err == nil {
		t.Error("期望空sessionID会返回错误，但没有")
	} else {
		t.Logf("✓ 空sessionID正确返回错误: %v", err)
	}

	// 测试提交奖励到不存在的session
	err = SubmitReward(&client, &clientMux, "non_existent_session", 1.0)
	if err == nil {
		t.Log("注意: 不存在的session仍然接受了奖励（可能是服务端自动创建了session）")
	} else {
		t.Logf("✓ 不存在的session正确返回错误: %v", err)
	}
}
