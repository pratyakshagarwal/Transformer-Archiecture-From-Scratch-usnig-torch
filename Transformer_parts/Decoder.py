import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#########################################################################################################################################################3
class DecoderSelfAttention(nn.Module):
    """
    Decoder self-attention module used in Transformer architectures.

    Args:
        config (object): Configuration object containing:
            - n_embd (int): Embedding size.
            - n_head (int): Number of attention heads.
            - dropout (float): Dropout probability.
            - bias (bool): Whether to include bias in linear layers.
            - max_target_len (int): Maximum length of target sequences.

    Attributes:
        attn (nn.Linear): Linear layer for computing attention scores.
        proj (nn.Linear): Linear layer for projecting attention outputs.
        attn_dropout (nn.Dropout): Dropout layer for attention scores.
        proj_dropout (nn.Dropout): Dropout layer for projected outputs.
        n_head (int): Number of attention heads.
        n_embd (int): Embedding size.
        dropout (float): Dropout probability.
        bias (torch.Tensor): Upper triangular mask for attention.

    Shape:
        - Input: (B, T, C)
        - Output: (B, T, C)

        - B: Batch size
        - T: Sequence length
        - C: Embedding dimension
    """

    def __init__(self, config):
        super(DecoderSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        # Linear layer for computing attention scores
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Linear layer for projecting attention outputs
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Upper triangular mask for attention
        self.register_buffer('bias', torch.tril(torch.ones(config.max_target_len, config.max_target_len)).view(1, 1, config.max_target_len, config.max_target_len))

    def forward(self, x):
        """
        Forward pass of the DecoderSelfAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape
        q, k, v = self.attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled Dot-Product Attention with mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(2, 1).contiguous().view(B, T, C)

        # Projection and dropout
        y = self.proj_dropout(self.proj(y))

        return y


class DecoderCrossAttention(nn.Module):
    """
    Decoder cross-attention module used in Transformer architectures.

    Args:
        config (object): Configuration object containing:
            - n_embd (int): Embedding size.
            - n_head (int): Number of attention heads.
            - dropout (float): Dropout probability.
            - bias (bool): Whether to include bias in linear layers.

    Attributes:
        attn_for_k_nd_v (nn.Linear): Linear layer for calculating keys and values.
        attn_for_q (nn.Linear): Linear layer for calculating queries.
        proj (nn.Linear): Linear layer for projecting attention outputs.
        attn_dropout (nn.Dropout): Dropout layer for attention scores.
        proj_dropout (nn.Dropout): Dropout layer for projected outputs.
        n_head (int): Number of attention heads.
        n_embd (int): Embedding size.
        dropout (float): Dropout probability.

    Shape:
        - Input: Tuple of two tensors (encoder_out, decoder_out):
            - encoder_out: (B, T, C)
            - decoder_out: (B, T, C)
        - Output: (B, T, C)

        - B: Batch size
        - T: Sequence length
        - C: Embedding dimension
    """

    def __init__(self, config):
        super(DecoderCrossAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        # Linear layer for calculating keys and values and queries
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Linear layer for projecting attention outputs
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        """
        Forward pass of the DecoderCrossAttention module.

        Args:
            x (tuple): Tuple of two tensors (encoder_out, decoder_out).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        encoder_out, decoder_out = x[0], x[1]
        B, T, C = decoder_out.shape
        Be, Te, Ce = encoder_out.shape

        # Calculating queries, keys, and values
        q = self.query(decoder_out)
        k = self.key(encoder_out)
        v = self.value(encoder_out)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, Te, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, Te, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled Dot-Product Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(2, 1).contiguous().view(B, T, C)

        # Projection and dropout
        y = self.proj_dropout(self.proj(y))
        return y

class DecoderFeedForward(nn.Module):
    """
    Feed-forward module used in Transformer decoders.

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
        super(DecoderFeedForward, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass of the DecoderFeedForward module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        fc1_out = self.fc1(x)
        relu_out = self.relu(fc1_out)
        fc2_out = self.drop(self.fc2(relu_out))
        return fc2_out

class DecoderBlock(nn.Module):
    """
    Decoder block used in Transformer decoders.

    Args:
        config (object): Configuration object containing model parameters.

    Attributes:
        dsa (DecoderSelfAttention): Decoder self-attention module.
        dca (DecoderCrossAttention): Decoder cross-attention module.
        dff (DecoderFeedForward): Decoder feed-forward module.
        ln1 (nn.LayerNorm): Layer normalization for self-attention output.
        ln2 (nn.LayerNorm): Layer normalization for cross-attention output.
        ln3 (nn.LayerNorm): Layer normalization for feed-forward output.
    """

    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        self.dsa = DecoderSelfAttention(config)
        self.dca = DecoderCrossAttention(config)
        self.dff = DecoderFeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        """
        Forward pass of the DecoderBlock module.

        Args:
            x (tuple): Tuple of two tensors (encoder_out, decoder_out).

        Returns:
            tuple: Tuple containing two tensors:
                - encoder_out: Output from encoder (unchanged).
                - dff_out: Output tensor of shape (B, T, C) from decoder block.
        """
        encoder_out, decoder_out = x[0], x[1]
        # Decoder self-attention
        dsa_out = decoder_out + self.dsa(self.ln1(decoder_out))

        # Decoder cross-attention
        dca_out = dsa_out + self.dca((encoder_out, self.ln2(dsa_out)))

        # Decoder feed-forward
        dff_out = dca_out + self.dff(self.ln3(dca_out))

        return (encoder_out, dff_out)

class Decoder(nn.Module):
    """
    Transformer Decoder module.

    Args:
        config (object): Configuration object containing:
            - target_vocab_size (int): Size of the target vocabulary.
            - n_embd (int): Embedding size.
            - max_target_len (int): Maximum target sequence length.
            - n_layer (int): Number of decoder blocks.
            - dropout (float): Dropout probability.
            - device (torch.device): Device for the tensors.

    Attributes:
        dwte (nn.Embedding): Token Embedding layer.
        dwpe (nn.Embedding): Positional Embedding layer.
        h (nn.ModuleList): List of DecoderBlocks.
        drop (nn.Dropout): Dropout layer.
        ln_f (nn.LayerNorm): Final Layer Normalization.
    """

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.dwte = nn.Embedding(config.target_vocab_size, config.n_embd)
        self.dwpe = nn.Embedding(config.max_target_len, config.n_embd)
        self.h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        """
        Forward pass of the Decoder module.

        Args:
            x (tuple): Tuple of two tensors (encoder_out, decoder_out).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        encoder_out, decoder_out = x[0], x[1]
        b, t = decoder_out.shape
        token_emb = self.dwte(decoder_out)
        pos_emb = self.dwpe(torch.arange(t, device=decoder_out.device))
        out = self.drop(token_emb + pos_emb)
        x = (encoder_out, out)
        for block in self.h:
            x = block(x)
        out = self.ln_f(out)
        return out

if __name__ == '__main__':
    pass