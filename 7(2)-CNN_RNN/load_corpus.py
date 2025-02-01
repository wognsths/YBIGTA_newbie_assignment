from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    # 구현하세요!
    ''' Dataset 로드 후 텍스트 저장 '''
    dataset = load_dataset("google-research-datasets/poem_sentiment", split="train")
    for data in dataset:
        text = data["verse_text"]
        corpus.extend(text.strip().split())
    return corpus