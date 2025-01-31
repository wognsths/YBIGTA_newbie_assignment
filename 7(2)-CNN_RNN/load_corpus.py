from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    # 구현하세요!
    dataset = load_dataset("google-research-datasets/poem_sentiment", split="train")
    for example in dataset:
        text = example["verse_text"]
        corpus.extend(text.strip().split())
    return corpus