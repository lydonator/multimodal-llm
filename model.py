# model.py

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from typing import Optional

class MultimodalGPT2(nn.Module):
    def __init__(self, gpt2_model: GPT2LMHeadModel, audio_embedding_dim: int):
        super(MultimodalGPT2, self).__init__()
        self.gpt2 = gpt2_model

        # Projection layer to match GPT-2's hidden size
        self.audio_proj = nn.Linear(audio_embedding_dim, gpt2_model.config.n_embd)

        # Cross-Attention Layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=gpt2_model.config.n_embd,
            num_heads=gpt2_model.config.n_head,
            dropout=gpt2_model.config.attn_pdrop,
        )

        # Layer Norm
        self.layer_norm = nn.LayerNorm(gpt2_model.config.n_embd, eps=gpt2_model.config.layer_norm_epsilon)

        # Dropout
        self.dropout = nn.Dropout(gpt2_model.config.resid_pdrop)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
    ):
        # Get GPT-2 outputs
        outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        if audio_embeddings is not None:
            # Project audio embeddings to match GPT-2's hidden size
            audio_embeddings = self.audio_proj(audio_embeddings)  # (batch_size, hidden_size)

            # Add sequence dimension to audio embeddings
            audio_embeddings = audio_embeddings.unsqueeze(1)  # (batch_size, 1, hidden_size)

            # Transpose for multihead attention (required shape: seq_len, batch_size, hidden_size)
            hidden_states = hidden_states.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
            audio_embeddings = audio_embeddings.transpose(0, 1)  # (1, batch_size, hidden_size)

            # Cross-Attention: Text attends to Audio
            attn_output, _ = self.cross_attention(
                query=hidden_states,
                key=audio_embeddings,
                value=audio_embeddings,
            )

            # Residual connection and layer normalization
            hidden_states = hidden_states + self.dropout(attn_output)
            hidden_states = self.layer_norm(hidden_states)

            # Transpose back to (batch_size, seq_len, hidden_size)
            hidden_states = hidden_states.transpose(0, 1)

        # Pass through LM Head
        lm_logits = self.gpt2.lm_head(hidden_states)  # (batch_size, seq_len, vocab_size)

        return {'logits': lm_logits}
