import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)))
class EncoderSelfAttention(nn.Module):
    """
    Encoder self-attention module used in Transformer architectures.

    Args:
        config (object): Configuration object containing:
            - n_embd (int): Embedding size.
            - n_head (int): Number of attention heads.
            - dropout (float): Dropout probability.
            - bias (bool): Whether to include bias in linear layers.

    Attributes:
        attn (nn.Linear): Linear layer for computing attention scores.
        proj (nn.Linear): Linear layer for projecting attention outputs.
        attn_drop (nn.Dropout): Dropout layer for attention scores.
        proj_drop (nn.Dropout): Dropout layer for projected outputs.
        n_head (int): Number of attention heads.
        dropout (float): Dropout probability.
        n_embd (int): Embedding size.

    Shape:
        - Input: (B, T, C)
        - Output: (B, T, C)

        - B: Batch size
        - T: Sequence length
        - C: Embedding dimension
    """

    def __init__(self, config):
        super(EncoderSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        # Linear layer for computing attention scores
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Linear layer for projecting attention outputs
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout layers
        self.attn_drop = nn.Dropout(config.dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.dropout = config.dropout
        self.n_embd = config.n_embd

    def forward(self, x):
        """
        Forward pass of the EncoderSelfAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape
        q, k, v = self.attn(x).split(C, dim=-1)

        # Reshape and transpose for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = self.attn_drop(F.softmax(attn, dim=-1))

        # Weighted sum
        y = attn @ v
        y = y.transpose(1, 2).reshape(B, T, C)

        # Projection and dropout
        y = self.proj_drop(self.proj(y))
        return y

class EncoderFeedForward(nn.Module):
    """
    Feed-forward module used in Transformer encoders.

    Args:
        config (object): Configuration object containing:
            - n_embd (int): Embedding size.
            - dropout (float): Dropout probability.

    Attributes:
        fc1 (nn.Linear): First linear layer.
        gelu (nn.GELU): GELU activation function.
        fc2 (nn.Linear): Second linear layer.
        drop (nn.Dropout): Dropout layer.
    """

    def __init__(self, config):
        super(EncoderFeedForward, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass of the EncoderFeedForward module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        fc1_out = self.fc1(x)
        relu_out = self.relu(fc1_out)
        fc2_out = self.drop(self.fc2(relu_out))
        return fc2_out


class EncoderBlock(nn.Module):
    """
    Encoder block in a Transformer architecture.

    Args:
        config (object): Configuration object containing:
            - n_embd (int): Embedding size.
            - n_head (int): Number of attention heads.
            - dropout (float): Dropout probability.
            - bias (bool): Whether to include bias in linear layers.

    Attributes:
        esa (EncoderSelfAttention): Encoder Self-Attention module.
        eff (EncoderFeedForward): Encoder Feed-Forward module.
        ln1 (nn.LayerNorm): Layer normalization after the first self-attention.
        ln2 (nn.LayerNorm): Layer normalization after the feed-forward layer.
    """

    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.esa = EncoderSelfAttention(config)
        self.eff = EncoderFeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        """
        Forward pass of the EncoderBlock module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        # Self-Attention Layer Normalization
        x = x + self.esa(self.ln1(x))

        # Feed-Forward Layer Normalization
        x = x + self.eff(self.ln2(x))
        return x

class Encoder(nn.Module):
    """
    Transformer Encoder module.

    Args:
        config (object): Configuration object containing:
            - input_vocab_size (int): Size of the input vocabulary.
            - max_input_len (int): Maximum input sequence length.
            - n_embd (int): Embedding size.
            - n_layer (int): Number of encoder blocks.
            - dropout (float): Dropout probability.
            - device (torch.device): Device for the tensors.

    Attributes:
        ewte (nn.Embedding): Token Embedding layer.
        ewpe (nn.Embedding): Positional Embedding layer.
        drop (nn.Dropout): Dropout layer.
        h (nn.ModuleList): List of EncoderBlocks.
        ln_f (nn.LayerNorm): Final Layer Normalization.
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.ewte = nn.Embedding(config.input_vocab_size, config.n_embd)
        self.ewpe = nn.Embedding(config.max_input_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        """
        Forward pass of the Encoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        b, t = x.shape
        tok_emb = self.ewte(x)
        pos_emb = self.ewpe(torch.arange(t, device=self.config.device))
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x


if __name__ == '__main__':
    pass