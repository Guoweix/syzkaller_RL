#!/usr/bin/env python3
"""
A3C (Asynchronous Actor-Critic) 模型实现
基于 SyscallAttentionEncoder 进行状态编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
import numpy as np
import threading
import time
from collections import deque
from typing import Dict, List, Tuple, Optional
from syscall_attention import SyscallAttentionEncoder


class A3CNetwork(nn.Module):
    """A3C 网络结构，包含 Actor 和 Critic"""
    
    def __init__(
        self,
        num_syscalls: int = 8056,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_actions: int = 5,
        hidden_dim: int = 256,
        param_dim: int = 50
    ):
        super(A3CNetwork, self).__init__()
        
        # Syscall序列编码器
        self.syscall_encoder = SyscallAttentionEncoder(
            num_syscalls=num_syscalls,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # 额外状态特征（exec_time, error_count等）
        self.state_feature_dim = 4  # call_count, exec_time, error_count, sequence_length
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim + self.state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor 网络 (输出动作概率)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        # 参数网络 (输出动作参数概率, 针对action_type==0)
        self.param_head_type0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, param_dim)
        )
        # 第三参数网络 (针对action_type==2)
        self.param_head_type2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, param_dim)
        )
        # 第二参数网络 (针对action_type==3)
        self.param_head_type3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, param_dim)
        )
        
        # Critic 网络 (输出状态价值)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.num_actions = num_actions
        self.param_dim = param_dim
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 包含call_sequence和其他状态信息的字典
            
        Returns:
            action_logits: 动作概率的logits
            param_logits: 参数概率的logits
            value: 状态价值
        """
        # 编码系统调用序列
        call_sequence = torch.tensor(state['call_sequence'], dtype=torch.long).unsqueeze(0)
        if call_sequence.size(1) == 0:
            # 如果序列为空，创建一个padding token
            call_sequence = torch.zeros(1, 1, dtype=torch.long)
        
        syscall_features = self.syscall_encoder(call_sequence)  # (1, embed_dim)
        
        # 提取其他状态特征
        other_features = torch.tensor([
            state.get('call_count', 0),
            state.get('exec_time', 0) / 1000.0,  # 归一化执行时间
            state.get('error_count', 0),
            len(state['call_sequence'])  # 序列长度
        ], dtype=torch.float32).unsqueeze(0)  # (1, state_feature_dim)
        
        # 融合特征
        combined_features = torch.cat([syscall_features, other_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # 获取动作概率和状态价值
        action_logits = self.actor(fused_features)
        param_logits_type0 = self.param_head_type0(fused_features)
        param_logits_type2 = self.param_head_type2(fused_features)
        param_logits_type3 = self.param_head_type3(fused_features)
        value = self.critic(fused_features)
        
        return action_logits, param_logits_type0, param_logits_type2, param_logits_type3, value
    
    def get_action_and_value(self, state: Dict) -> Tuple[int, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据状态获取动作和价值
        
        Returns:
            action: 选择的动作
            action_log_prob: 动作的对数概率
            value: 状态价值
            param_probs: 参数概率分布
        """
        with torch.no_grad():
            action_logits, param_logits_type0, param_logits_type2, param_logits_type3, value = self.forward(state)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            param_probs_type0 = F.softmax(param_logits_type0, dim=-1)
            param_probs_type2 = F.softmax(param_logits_type2, dim=-1)
            param_probs_type3 = F.softmax(param_logits_type3, dim=-1)
        
        return action.item(), action_log_prob, value.squeeze(), param_probs_type0.squeeze(0), param_probs_type2.squeeze(0), param_probs_type3.squeeze(0)


class A3CAgent:
    """A3C 智能体"""
    
    def __init__(
        self,
        global_model: A3CNetwork,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        epsilon_start: float = 0.1,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 100000,
        temp_start: float = 1.0,
        temp_end: float = 0.3,
        temp_decay_steps: int = 100000
    ):
        self.global_model = global_model
        self.local_model = A3CNetwork(
            num_syscalls=global_model.syscall_encoder.embedding.num_embeddings,
            embed_dim=global_model.syscall_encoder.embedding.embedding_dim,
            num_actions=global_model.num_actions,
            param_dim=global_model.param_dim
        )
        self.optimizer = optimizer
        
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        # 探索相关参数
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temp_decay_steps = max(1, temp_decay_steps)
        self.total_action_calls = 0
        
        # 同步本地模型
        self.sync_with_global()
        
    def sync_with_global(self):
        """同步本地模型与全局模型"""
        self.local_model.load_state_dict(self.global_model.state_dict())
    
    def _current_epsilon(self) -> float:
        ratio = min(1.0, self.total_action_calls / self.epsilon_decay_steps)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * ratio

    def _current_temperature(self) -> float:
        ratio = min(1.0, self.total_action_calls / self.temp_decay_steps)
        return self.temp_start + (self.temp_end - self.temp_start) * ratio

    def get_action(self, state: Dict) -> Tuple[int, Dict[str, List[float]]]:
        """
        根据状态获取动作
        
        Returns:
            action_type: 动作类型 (0-4)
            param_probs: 参数概率分布（长度为 param_dim）
        """
        with torch.no_grad():
            action_logits, param_logits_type0, param_logits_type2, param_logits_type3, value = self.local_model.forward(state)
            temperature = self._current_temperature()
            scaled_logits = action_logits / max(1e-6, temperature)
            action_probs = F.softmax(scaled_logits, dim=-1)
            dist = Categorical(action_probs)
            self.total_action_calls += 1

            epsilon = self._current_epsilon()
            if np.random.rand() < epsilon:
                action_type = np.random.randint(0, self.local_model.num_actions)
            else:
                action_type = dist.sample().item()

            # 参数概率遮罩: 仅在 action_type == 0 时返回有效参数概率，否则返回空列表
            call_count = int(state.get('call_count', 0))
            seq_len = len(state.get('call_sequence', []))
            valid_len = max(0, min(call_count, seq_len, self.local_model.param_dim))
            if action_type == 0:
                if valid_len == 0:
                    param_probs_list = []
                else:
                    masked_logits = self._mask_param_logits(param_logits_type0.squeeze(0), valid_len)
                    probs = F.softmax(masked_logits, dim=-1)[:valid_len]
                    param_probs_list = probs.detach().cpu().numpy().tolist()
                return action_type, {"type0": param_probs_list, "type2": [], "type3": []}
            elif action_type == 2:
                # mask 范围使用 call_count+1
                valid_len2 = max(0, min(call_count + 1, seq_len, self.local_model.param_dim))
                if valid_len2 == 0:
                    param_probs_list2 = []
                else:
                    masked_logits2 = self._mask_param_logits(param_logits_type2.squeeze(0), valid_len2)
                    probs2 = F.softmax(masked_logits2, dim=-1)[:valid_len2]
                    param_probs_list2 = probs2.detach().cpu().numpy().tolist()
                return action_type, {"type0": [], "type2": param_probs_list2, "type3": []}
            elif action_type == 3:
                if valid_len == 0:
                    param_probs_list3 = []
                else:
                    masked_logits3 = self._mask_param_logits(param_logits_type3.squeeze(0), valid_len)
                    probs3 = F.softmax(masked_logits3, dim=-1)[:valid_len]
                    param_probs_list3 = probs3.detach().cpu().numpy().tolist()
                return action_type, {"type0": [], "type2": [], "type3": param_probs_list3}
            else:
                return action_type, {"type0": [], "type2": [], "type3": []}
    
    def compute_loss(
        self,
        states: List[Dict],
        actions: List[int],
        action_params: List[int],
        rewards: List[float],
        next_value: float = 0.0
    ) -> torch.Tensor:
        """
        计算 A3C 损失
        
        Args:
            states: 状态序列
            actions: 动作序列
            action_params: 动作参数序列
            rewards: 奖励序列
            next_value: 下一个状态的价值
            
        Returns:
            总损失
        """
        if not states:
            return torch.tensor(0.0, dtype=torch.float32)
        
        device = next(self.local_model.parameters()).device
        
        # 计算累积奖励
        returns = []
        R = next_value
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        
        policy_losses = []
        value_losses = []
        entropies = []
        param_policy_losses = []
        param_entropies = []
        
        for i, (state, action, param) in enumerate(zip(states, actions, action_params)):
            action_logits, param_logits_type0, param_logits_type2, param_logits_type3, value = self.local_model.forward(state)
            action_logits = action_logits.to(device)
            param_logits_type0 = param_logits_type0.to(device)
            param_logits_type2 = param_logits_type2.to(device)
            param_logits_type3 = param_logits_type3.to(device)
            value = value.squeeze().to(device)
            
            # 计算优势
            advantage = returns[i] - value
            
            # 动作策略损失
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = Categorical(action_probs)
            action_tensor = torch.tensor(action, dtype=torch.long, device=device)
            action_log_prob = action_dist.log_prob(action_tensor)
            policy_losses.append(-action_log_prob * advantage.detach())
            entropies.append(action_dist.entropy())
            
            # 状态价值损失
            value_losses.append(F.mse_loss(value, returns[i]))
            
            # 参数策略损失（仅在动作为0时生效）
            if action in (0,2,3):
                call_count = int(state.get('call_count', 0))
                seq_len = len(state.get('call_sequence', []))
                if action == 2:
                    raw_valid = call_count + 1
                else:
                    raw_valid = call_count
                valid_len = max(1, min(raw_valid, seq_len, self.local_model.param_dim))
                if action == 0:
                    target_logits = param_logits_type0
                elif action == 2:
                    target_logits = param_logits_type2
                else:
                    target_logits = param_logits_type3
                masked_param_logits = self._mask_param_logits(target_logits, valid_len)
                param_probs = F.softmax(masked_param_logits, dim=-1).squeeze(0)
                param_dist = Categorical(param_probs)
                param_idx = max(0, min(param, valid_len - 1))
                param_tensor = torch.tensor(param_idx, dtype=torch.long, device=device)
                param_log_prob = param_dist.log_prob(param_tensor)
                param_policy_losses.append(-param_log_prob * advantage.detach())
                param_entropies.append(param_dist.entropy())
        
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy_bonus = torch.stack(entropies).mean()
        
        if param_policy_losses:
            param_policy_loss = torch.stack(param_policy_losses).mean()
            param_entropy_bonus = torch.stack(param_entropies).mean()
        else:
            param_policy_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            param_entropy_bonus = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        total_loss = (
            policy_loss +
            param_policy_loss +
            self.value_loss_coef * value_loss -
            self.entropy_coef * (entropy_bonus + param_entropy_bonus)
        )
        
        return total_loss
    
    @staticmethod
    def _mask_param_logits(param_logits: torch.Tensor, valid_len: int) -> torch.Tensor:
        """根据有效长度(valid_len)屏蔽无效的参数位置"""
        masked = param_logits.clone()
        param_dim = masked.size(-1)
        valid_len = max(1, min(valid_len, param_dim))
        if param_logits.dim() == 1:
            if valid_len < param_dim:
                masked[valid_len:] = -1e9
        else:
            if valid_len < param_dim:
                masked[:, valid_len:] = -1e9
        return masked

    def update_global_model(
        self,
        states: List[Dict],
        actions: List[int],
        action_params: List[int],
        rewards: List[float],
        next_value: float = 0.0
    ):
        """更新全局模型"""
        # 计算损失
        loss = self.compute_loss(states, actions, action_params, rewards, next_value)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.max_grad_norm)
        
        # 将本地梯度复制到全局模型
        for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
            if local_param.grad is not None:
                global_param.grad = local_param.grad.clone()
        
        # 更新全局模型
        self.optimizer.step()
        
        # 重新同步本地模型
        self.sync_with_global()
        
        return loss.item()


class ExperienceBuffer:
    """经验缓冲区"""
    
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.action_params = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
    
    def add(self, state: Dict, action: int, action_param: int, reward: float):
        """添加经验"""
        self.states.append(state)
        self.actions.append(action)
        self.action_params.append(action_param)
        self.rewards.append(reward)
    
    def get_batch(self) -> Tuple[List[Dict], List[int], List[int], List[float]]:
        """获取所有经验"""
        return list(self.states), list(self.actions), list(self.action_params), list(self.rewards)
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.action_params.clear()
        self.rewards.clear()
    
    def __len__(self):
        return len(self.states)


class A3CTrainer:
    """A3C 训练器"""
    
    def __init__(
        self,
        num_syscalls: int = 8056,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_actions: int = 5,
        hidden_dim: int = 256,
        param_dim: int = 50,
        lr: float = 3e-4,
        gamma: float = 0.99,
        update_interval: int = 20,
        save_interval: int = 1000,
        # 探索参数（可按需调参）
        epsilon_start: float = 0.1,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 200000,
        temp_start: float = 1.0,
        temp_end: float = 0.4,
        temp_decay_steps: int = 200000
    ):
        # 创建全局模型
        self.global_model = A3CNetwork(
            num_syscalls=num_syscalls,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            param_dim=param_dim
        )
        
        # 共享优化器
        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=lr)
        
        # 创建智能体
        self.agent = A3CAgent(
            global_model=self.global_model,
            optimizer=self.optimizer,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            temp_start=temp_start,
            temp_end=temp_end,
            temp_decay_steps=temp_decay_steps
        )
        
        # 训练参数
        self.update_interval = update_interval
        self.save_interval = save_interval
        
        # 统计信息
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_losses = deque(maxlen=100)
        
        # 会话管理
        self.session_buffers = {}  # session_id -> ExperienceBuffer
        self.session_states = {}   # session_id -> last_state
        
        print(f"A3C Trainer initialized with {sum(p.numel() for p in self.global_model.parameters()):,} parameters")
    
    def get_action(self, session_id: str, state: Dict) -> Tuple[int, List[float]]:
        """为特定会话获取动作及参数概率"""
        self.session_states[session_id] = state
        action_type, param_probs = self.agent.get_action(state)
        # self.agent.get_action 现在可能直接返回 list (action_type!=0 时为空列表)
        if isinstance(param_probs, np.ndarray):
            param_probs = param_probs.tolist()
        return action_type, param_probs
    
    def submit_reward(
        self,
        session_id: str,
        reward: float,
        action: int,
        action_param: int = 0
    ) -> Optional[float]:
        """提交奖励并可能触发更新，包含动作参数"""
        # 确保会话缓冲区存在
        if session_id not in self.session_buffers:
            self.session_buffers[session_id] = ExperienceBuffer()
        
        buffer = self.session_buffers[session_id]
        
        # 添加经验（需要之前的状态）
        if session_id in self.session_states:
            buffer.add(self.session_states[session_id], action, action_param, reward)
            self.total_steps += 1
        
        loss = None
        
        # 检查是否需要更新
        if len(buffer) >= self.update_interval:
            states, actions, params, rewards = buffer.get_batch()
            
            # 计算下一个状态的价值（如果有的话）
            next_value = 0.0
            if session_id in self.session_states:
                with torch.no_grad():
                    _, _, _, _, value = self.agent.local_model.forward(self.session_states[session_id])
                    next_value = value.squeeze().item()
            
            # 更新模型
            loss = self.agent.update_global_model(states, actions, params, rewards, next_value)
            
            # 清空缓冲区
            buffer.clear()
            
            # 记录统计信息
            self.episode_count += 1
            self.episode_rewards.append(sum(rewards))
            self.episode_losses.append(loss)
            
            # 保存模型
            if self.episode_count % self.save_interval == 0:
                self.save_model(f"a3c_model_episode_{self.episode_count}.pth")
            
            # 打印统计信息
            if self.episode_count % 10 == 0:
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                avg_loss = np.mean(self.episode_losses) if self.episode_losses else 0
                print(f"Episode {self.episode_count}, Avg Reward: {avg_reward:.3f}, "
                      f"Avg Loss: {avg_loss:.6f}, Total Steps: {self.total_steps}")
        
        return loss
    
    def end_session(self, session_id: str):
        """结束会话，清理资源"""
        if session_id in self.session_buffers:
            buffer = self.session_buffers[session_id]
            if len(buffer) > 0:
                # 处理剩余的经验
                states, actions, params, rewards = buffer.get_batch()
                loss = self.agent.update_global_model(states, actions, params, rewards, 0.0)
                
                self.episode_count += 1
                self.episode_rewards.append(sum(rewards))
                if loss is not None:
                    self.episode_losses.append(loss)
            
            del self.session_buffers[session_id]
        
        if session_id in self.session_states:
            del self.session_states[session_id]
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint.get('episode_count', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        
        # 重新同步agent
        self.agent.sync_with_global()
        print(f"Model loaded from {path}")
    
    def get_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "active_sessions": len(self.session_buffers),
            "avg_reward_last_100": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "avg_loss_last_100": np.mean(self.episode_losses) if self.episode_losses else 0
        }


# 全局训练器实例
global_trainer = None

def get_trainer() -> A3CTrainer:
    """获取全局训练器实例"""
    global global_trainer
    if global_trainer is None:
        global_trainer = A3CTrainer(update_interval=1)
    return global_trainer


if __name__ == "__main__":
    # 简单测试
    trainer = A3CTrainer()
    
    # 模拟一些训练数据
    for episode in range(5):
        session_id = f"test_session_{episode}"
        
        for step in range(10):
            # 模拟状态
            state = {
                'session_id': session_id,
                'call_sequence': np.random.randint(0, 100, size=np.random.randint(5, 20)).tolist(),
                'call_count': step,
                'exec_time': np.random.randint(10, 1000),
                'error_count': np.random.randint(0, 3)
            }
            
            # 获取动作
            action_type, param_probs = trainer.get_action(session_id, state)
            if action_type == 0:
                param_array = np.asarray(param_probs, dtype=np.float32)
                if param_array.size > 0:
                    valid_len = max(1, min(len(state['call_sequence']), param_array.size))
                    mask = np.zeros_like(param_array)
                    mask[:valid_len] = 1.0
                    masked_probs = param_array * mask
                    if masked_probs.sum() <= 0:
                        action_param = int(np.argmax(param_array[:valid_len]))
                    else:
                        action_param = int(np.argmax(masked_probs))
                else:
                    action_param = 0
            else:
                action_param = action_type % 3
            
            # 模拟奖励
            reward = np.random.random() * 2 - 1  # -1 到 1 之间的随机奖励
            
            # 提交奖励
            loss = trainer.submit_reward(session_id, reward, action_type, action_param)
            
            if loss is not None:
                print(f"Episode {episode}, Step {step}, Loss: {loss:.6f}")
        
        # 结束会话
        trainer.end_session(session_id)
    
    print("Training test completed!")
    print(f"Final stats: {trainer.get_stats()}")