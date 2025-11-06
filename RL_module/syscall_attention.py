import torch
import torch.nn as nn
import torch.nn.functional as F

class SyscallAttentionEncoder(nn.Module):
    def __init__(self, num_syscalls=8056, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_syscalls, embed_dim)
        self.pos_embedding = nn.Embedding(512, embed_dim)  # 支持最长 512 长度
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, syscall_seq):
        """
        syscall_seq: (B, L) 整数序列
        输出: (B, D) 序列向量（取均值或 CLS token）
        """
        B, L = syscall_seq.shape
        pos_ids = torch.arange(L, device=syscall_seq.device).unsqueeze(0).expand(B, L)
        x = self.embedding(syscall_seq) + self.pos_embedding(pos_ids)
        x = self.encoder(x)  # (B, L, D)
        # 可选：池化成固定维度输出
        context = x.mean(dim=1)  # 或者取 x[:, 0, :]
        return context

