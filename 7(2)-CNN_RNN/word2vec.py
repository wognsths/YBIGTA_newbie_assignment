import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!

        if isinstance(corpus[0], str):
            encoded_corpus = []
            for token in corpus:
                ids = tokenizer.convert_tokens_to_ids(token)
                encoded_corpus.append(ids)
        else:
            encoded_corpus = corpus

        for epoch in range(num_epochs):
            if self.method == "cbow":
                self._train_cbow(encoded_corpus, optimizer, criterion)
            else:
                self._train_skipgram(encoded_corpus, optimizer, criterion)

            print(f"Epoch {epoch+1} done.")

    def _train_cbow(
        self,
        # 구현하세요!
        encoded_corpus: list[str],
        optimizer: Adam,
        criterion: nn.CrossEntropyLoss
    ) -> None:
        # 구현하세요!
        self.train()

        for center_index in range(self.window_size, len(encoded_corpus) - self.window_size):
            center_word_id = encoded_corpus[center_index]
            context_ids = []
            for t in range(-self.window_size, self.window_size + 1):
                if t == 0:
                    continue
                context_ids.append(encoded_corpus[center_index + t])
            context_ids_t = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0)

            context_embeds = self.embeddings(context_ids_t)
            cbow_embed = context_embeds.mean(dim=1)

            logits = self.weight(cbow_embed)

            center_word_id_t = torch.tensor([center_word_id], dtype=torch.long)
            loss = criterion(logits, center_word_id_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        

    def _train_skipgram(
        self,
        # 구현하세요!
        encoded_corpus: list[int],
        optimizer: Adam,
        criterion: nn.CrossEntropyLoss
    ) -> None:
        self.train()

        for center_idx in range(self.window_size, len(encoded_corpus) - self.window_size):
            center_word_id = encoded_corpus[center_idx]

            # 중심 단어 임베딩 (shape: (1, d_model))
            center_word_id_t = torch.tensor([center_word_id], dtype=torch.long)
            center_embed = self.embeddings(center_word_id_t)

            # 주변(맥락) 단어 리스트
            context_ids = []
            for t in range(-self.window_size, self.window_size + 1):
                if t == 0:
                    continue
                context_ids.append(encoded_corpus[center_idx + t])

            # Skip-Gram은 주변 단어 각각에 대해 예측 -> 손실 합산
            total_loss = 0.0
            for ctx_id in context_ids:
                # (1, d_model) -> (1, vocab_size)
                logits = self.weight(center_embed)
                ctx_word_id_t = torch.tensor([ctx_id], dtype=torch.long)
                loss = criterion(logits, ctx_word_id_t)
                total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    # 구현하세요!
    pass