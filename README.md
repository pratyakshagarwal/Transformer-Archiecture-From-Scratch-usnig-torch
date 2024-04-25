This repository contains PyTorch implementation of a Transformer model for sequence-to-sequence tasks, particularly designed for machine translation. The Transformer architecture, introduced in the paper "Attention is All You Need" by Vaswani et al., has shown remarkable performance in various natural language processing tasks.

## **Transformer Model Components**

1. **Encoder**: The encoder module consists of multiple layers of encoder blocks. Each encoder block includes a self-attention mechanism followed by a feed-forward neural network. The encoder processes the input sequence and generates a series of context-aware representations.

2. **Decoder**: The decoder module also comprises several layers of decoder blocks. Each decoder block incorporates self-attention, cross-attention (with the encoder output), and feed-forward layers. The decoder generates the output sequence by attending to both the input sequence (through the cross-attention mechanism) and the previously generated tokens.

3. **EncoderSelfAttention**: Implements self-attention mechanism for the encoder. It computes attention scores between different positions of the input sequence and aggregates information accordingly.

4. **EncoderFeedForward**: Consists of feed-forward neural network layers for the encoder blocks. It processes the output of the self-attention layer and produces context-aware representations.

5. **EncoderBlock**: Combines EncoderSelfAttention and EncoderFeedForward modules along with layer normalization to form an encoder block.
DecoderSelfAttention: Handles self-attention mechanism for the decoder. It computes attention scores within the decoder input sequence and applies masking to prevent attending to future tokens.

6. **DecoderCrossAttention**: Implements cross-attention mechanism for the decoder. It attends to the encoder output to incorporate contextual information from the input sequence.

7. **DecoderFeedForward**: Contains feed-forward neural network layers for the decoder blocks. It processes the output of the cross-attention layer and generates predictions for the next tokens.

8. **DecoderBlock**: Integrates DecoderSelfAttention, DecoderCrossAttention, and DecoderFeedForward modules along with layer normalization to form a decoder block.

9. **Transformer**: The main Transformer model, which combines the encoder and decoder modules. It takes input sequences, processes them through the encoder-decoder architecture, and produces output sequences.

### Usage
To use the Transformer model for sequence-to-sequence tasks, follow these steps:

- Instantiate the Transformer model with appropriate configuration parameters.
- Prepare input sequences and target sequences (for training).
- Pass the input sequences through the encoder.
- Generate predictions for the target sequences using the decoder.
- Optionally, apply softmax to obtain probability distributions over the target vocabulary.

### Test Case
```bash

@dataclass
class Config:
    batch_size: int = 64
    max_input_len: int = 256
    max_target_len: int = 300
    input_vocab_size: int = 500 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    target_vocab_size: int = 800
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device = 'cpu'
        
config = Config()
transformer = Transformer(config)
    
encoder_idx = torch.randint(high=config.input_vocab_size, size=(config.batch_size, config.max_input_len), dtype=torch.int32)
print('encoder input shape:', encoder_idx.shape)
    
decoder_idx = torch.randint(high=config.target_vocab_size, size=(config.batch_size, config.max_target_len), dtype=torch.int32)
print('decoder block input shape:', decoder_idx.shape)
    
print(transformer((encoder_idx, decoder_idx), logits=False).shape)
```
### output
```bash
encoder input shape: torch.Size([64, 256])
decoder block input shape: torch.Size([64, 300])
torch.Size([64, 300, 800])
```

### Requirements
- Python 3.x
- PyTorch
- NumPy

### Acknowledgements
This implementation and test case are adapted from the Transformer architecture proposed by **Vaswani et al. (2017)** and incorporate design choices from various sources in the research community.

### References
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). **Attention is all you need**. In Advances in neural information processing systems (pp. 5998-6008).