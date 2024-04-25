import torch 
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    """
    Transformer model composed of an Encoder and a Decoder.

    Args:
        config (object): Configuration object containing:
            - n_embd (int): Embedding size.
            - target_vocab_size (int): Size of the target vocabulary.

    Attributes:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        fc1 (nn.Linear): Final linear layer for output.
    """

    def __init__(self, config):
        super(Transformer, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.fc1 = nn.Linear(config.n_embd, config.target_vocab_size)

    def forward(self, x, logits=True):
        """
        Forward pass of the Transformer model.

        Args:
            x (tuple): Tuple containing:
                - encoder_idx (torch.Tensor): Input tensor for the encoder of shape (B, T).
                - decoder_in (torch.Tensor): Input tensor for the decoder of shape (B, T_dec).

            logits (bool): If True, returns logits; if False, returns probabilities after softmax.

        Returns:
            torch.Tensor: Output tensor of shape (B, T_dec, target_vocab_size).
        """
        encoder_idx, decoder_in = x[0], x[1]

        # Encode
        encoder_out = self.encoder(encoder_idx)

        # Decode
        decoder_out = self.decoder((encoder_out, decoder_in))

        # Final linear layer
        probs = self.fc1(decoder_out)

        # Optionally apply softmax
        if not logits:
            probs = F.softmax(probs, dim=-1)

        return probs


if __name__ == '__main__':
    pass