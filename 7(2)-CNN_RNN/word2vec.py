import random

import torch
from torch import nn, Tensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"],
        device="cpu"
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False) # y = x @ W.T (batch_size, d_model) * (d_model, vocab_size)
        self.window_size = window_size
        self.method = method
        # 구현하세요!

        self.device = device
        self.to(device)

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach() # Gradient 추적을 끊기 위한 방법 -> 학습 이후 모델의 임베딩 값을 가져올 수 있다.

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int,
        batch_size: int = 256
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)

        # 만약 corpus가 문자열 리스트라면 tokenizer로 정수 ID로 변환
        if isinstance(corpus[0], str):
            encoded_corpus = tokenizer.convert_tokens_to_ids(corpus)
        else:
            encoded_corpus = corpus

        for epoch in range(num_epochs):
            if self.method == "cbow":
                self._train_cbow(encoded_corpus, optimizer, criterion, batch_size)
            else:
                self._train_skipgram(encoded_corpus, optimizer, criterion, batch_size)

            print(f"Epoch {epoch+1}/{num_epochs} done.")

    def _train_skipgram(
        self,
        encoded_corpus: list[int],
        optimizer: Adam,
        criterion: nn.CrossEntropyLoss,
        batch_size: int
    ) -> None:
        '''Skip-gram을 미니배치로 학습'''
        self.train()

        # (1) (center, context) 쌍 전부 생성
        # widow_size = 2, corpus = ['the', 'cat', 'is', 'on', 'the', 'tree'] 인 경우
        # 토큰화 후 [0, 1, 2, 3, 0, 4]
        # ex: 중심 단어 2 (is) -> 주변 단어 0 (the), 1 (cat), 3 (on), 0 (the)
        # pairs = [(2, 0), (2, 1), (2, 3), (2, 0)] 
        pairs: list[tuple] = []
        for center_idx in range(self.window_size, len(encoded_corpus) - self.window_size):
            center_word_id = encoded_corpus[center_idx]
            # 주변 단어들
            for t in range(-self.window_size, self.window_size + 1):
                if t == 0:
                    continue
                context_word_id = encoded_corpus[center_idx + t]
                pairs.append((center_word_id, context_word_id))

        # (2) 셔플 -> 학습 다양성
        random.shuffle(pairs)

        # (3) 미니배치 단위로 학습
        total_loss = 0.0
        num_batches = (len(pairs) + batch_size - 1) // batch_size

        for i in range(num_batches):
            batch_data = pairs[i*batch_size:(i+1)*batch_size]
            if not batch_data:
                continue
            
            # batch_data = [(center, context), (center, context), ...]
            batch_centers = [p[0] for p in batch_data]  # shape: (B,)
            batch_contexts = [p[1] for p in batch_data] # shape: (B,)

            # 텐서 변환
            batch_centers_t = torch.tensor(batch_centers, dtype=torch.long, device=self.device)
            batch_contexts_t = torch.tensor(batch_contexts, dtype=torch.long, device=self.device)

            # (4) Foward
            center_embed = self.embeddings(batch_centers_t)   # shape: (B, d_model)
            logits = self.weight(center_embed)                # shape: (B, vocab_size)

            # (5) 손실 계산
            loss = criterion(logits, batch_contexts_t)

            # (6) Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[SkipGram] total_loss: {total_loss/num_batches:.4f}")

    def _train_cbow(
        self,
        encoded_corpus: list[int],
        optimizer: Adam,
        criterion: nn.CrossEntropyLoss,
        batch_size: int
    ) -> None:
        '''CBOW를 미니배치로 학습'''
        self.train()

        # (1) (context_list, center) 쌍 전부 생성
        # context_list 길이는 2*window_size, center는 1개
        cbow_data: list[tuple] = []
        for center_idx in range(self.window_size, len(encoded_corpus) - self.window_size):
            center_word_id = encoded_corpus[center_idx]
            context_ids = []
            for t in range(-self.window_size, self.window_size + 1):
                if t == 0:
                    continue
                context_ids.append(encoded_corpus[center_idx + t])
            # 한 예시 (context_ids, center)
            cbow_data.append((context_ids, center_word_id))

        # (2) 셔플
        random.shuffle(cbow_data)

        # (3) 미니배치로 학습
        total_loss = 0.0
        num_batches = (len(cbow_data) + batch_size - 1) // batch_size

        for i in range(num_batches):
            batch_data = cbow_data[i*batch_size:(i+1)*batch_size]
            if not batch_data:
                continue

            # batch_data = [([ctx1, ctx2, ...], center), (...), ...]
            batch_contexts = [item[0] for item in batch_data]  # shape: (B, 2*window_size)
            batch_centers = [item[1] for item in batch_data]   # shape: (B,)

            # 텐서 변환
            # (B, 2*window_size), (B,)
            batch_contexts_t = torch.tensor(batch_contexts, dtype=torch.long, device=self.device)
            batch_centers_t = torch.tensor(batch_centers, dtype=torch.long, device=self.device)

            # (4) 순전파: CBOW는 주변 단어 임베딩 평균
            # shape: (B, 2*window_size, d_model)
            context_embeds = self.embeddings(batch_contexts_t)
            # shape: (B, d_model)
            cbow_embed = context_embeds.mean(dim=1)

            # 출력 (B, vocab_size)
            logits = self.weight(cbow_embed)

            # (5) 손실 계산
            loss = criterion(logits, batch_centers_t)

            # (6) 역전파 & 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[CBOW] total_loss: {total_loss/num_batches:.4f}")