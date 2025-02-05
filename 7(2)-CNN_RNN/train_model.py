import torch
from torch import nn, optim, Tensor, LongTensor, FloatTensor
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

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
    #     requires_grad = True로 설정하여 학습 중에도 업데이트될 수 있도록 함
    model.embeddings.weight.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # (2) lr_scheduler 적용 (ReduceLROnPlateau) - validation f1이 개선되지 않으면 lr 감소
    #     F1 점수가 높아질수록 성능이 좋아진 것으로 판단
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=1, verbose=True
    )

    # load train, validation, test dataset
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True) # Shuffle=True로 일반화 성능 향상
    validation_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False)

    num_epochs = num_epochs

    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0
        for data in train_loader:
            optimizer.zero_grad()
            input_ids = tokenizer(data["verse_text"], padding=True, return_tensors="pt")\
                .input_ids.to(device)

            labels = data["label"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()

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
