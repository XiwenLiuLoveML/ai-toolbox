import torch
import torch.nn as nn

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target_word_idx):
        embed = self.embeddings(target_word_idx)  # [batch_size, embedding_dim]
        out = self.output(embed)                  # [batch_size, vocab_size]
        return out
