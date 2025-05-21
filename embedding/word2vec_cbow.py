import torch
import torch.nn as nn

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        embeds = self.embeddings(context_idxs)  # [batch_size, context_size, embedding_dim]
        context_vector = embeds.mean(dim=1)     # Average embeddings across context
        out = self.linear(context_vector)
        return out
