import torch
from torch import nn, optim, Tensor, LongTensor, FloatTensor
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

from tqdm import tqdm
from sklearn.metrics import f1_score

from word2vec import Word2Vec
from model import MyGRULanguageModel
from config import *

if __name__ == "__main__":
    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # load Word2Vec checkpoint and get trained embeddings
    word2vec = Word2Vec(vocab_size, d_model, window_size, method)
    checkpoint = torch.load("word2vec.pt")
    word2vec.load_state_dict(checkpoint)
    embeddings = word2vec.embeddings_weight()

    # declare model, criterion and optimizer
    model = MyGRULanguageModel(d_model, hidden_size, num_classes, embeddings).to(device)

    # (1) Embedding 레이어 Unfreeze 설정
    #     from_pretrained(... freeze=True)이더라도 아래와 같이 requires_grad를 True로 바꾸면 학습 가능
    model.embeddings.weight.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # (2) lr_scheduler 적용 (ReduceLROnPlateau) - validation f1이 개선되지 않으면 lr 감소
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=1, verbose=True
    )

    # load train, validation, test dataset
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False)

    # (3) num_epochs 조정 -> 예: 기존 10에서 20으로
    # config.py에 있는 값을 덮어쓰거나 직접 숫자 입력
    # num_epochs = 20
    num_epochs = num_epochs  # config.py값 그대로 쓸 수도 있음

    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0
        for data in train_loader:
            optimizer.zero_grad()
            input_ids = tokenizer(data["verse_text"], padding=True, return_tensors="pt")\
                .input_ids.to(device)

            labels = data["label"].to(device)

            # (4) 간단한 Dropout 예시 (임시):
            # input_ids를 임베딩한 후에 dropout을 적용하려면 model.forward 내부 수정이 필요하지만
            # 아래처럼 임시로 적용해볼 수도 있음 (구조가 다소 어색함)
            #
            # with torch.no_grad():
            #     embedded = model.embeddings(input_ids)
            # embedded = F.dropout(embedded, p=0.3, training=model.training)
            # logits = model.head(model.gru(embedded))

            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()

            # gradient clipping 예시 (선택)
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            loss_sum += loss.item()

        # Validation
        model.eval()
        preds = []
        val_labels = []
        with torch.no_grad():
            for data in validation_loader:
                input_ids = tokenizer(data["verse_text"], padding=True, return_tensors="pt")\
                    .input_ids.to(device)
                logits = model(input_ids)
                val_labels += data["label"].tolist()
                preds += logits.argmax(-1).cpu().tolist()

        macro_val = f1_score(val_labels, preds, average='macro')
        micro_val = f1_score(val_labels, preds, average='micro')

        # scheduler step
        scheduler.step(macro_val)

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {loss_sum/len(train_loader):.6f} | "
              f"Val MacroF1: {macro_val:.6f}, Val MicroF1: {micro_val:.6f}")

    # save model checkpoint
    torch.save(model.cpu().state_dict(), "checkpoint.pt")