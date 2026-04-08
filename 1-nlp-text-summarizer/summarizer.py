"""
NLP Text Summarizer using Transformer-based models (HuggingFace)
Supports extractive and abstractive summarization
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import numpy as np
import argparse
import textwrap

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ─────────────────────────────────────────────
# 1. EXTRACTIVE SUMMARIZER (TF-IDF / NLTK)
# ─────────────────────────────────────────────
class ExtractiveSummarizer:
    """Rank sentences by word-frequency score and pick the top-N."""

    def __init__(self, language: str = "english"):
        self.stop_words = set(stopwords.words(language))

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        words = [
            w.lower()
            for s in sentences
            for w in s.split()
            if w.lower() not in self.stop_words and w.isalpha()
        ]
        freq = FreqDist(words)
        max_freq = freq.most_common(1)[0][1]
        normalized = {w: f / max_freq for w, f in freq.items()}

        scores = []
        for sent in sentences:
            score = sum(normalized.get(w.lower(), 0) for w in sent.split())
            scores.append(score)

        top_idx = sorted(np.argsort(scores)[-num_sentences:])
        return " ".join(sentences[i] for i in top_idx)


# ─────────────────────────────────────────────
# 2. ABSTRACTIVE SUMMARIZER (T5 / BART)
# ─────────────────────────────────────────────
class AbstractiveSummarizer:
    """Use a seq2seq model to generate a brand-new summary."""

    MODEL_NAME = "facebook/bart-large-cnn"

    def __init__(self):
        print(f"Loading model: {self.MODEL_NAME} …")
        self.summarizer = pipeline("summarization", model=self.MODEL_NAME)
        print("Model ready.")

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 40,
    ) -> str:
        chunks = self._chunk(text, max_tokens=900)
        summaries = []
        for chunk in chunks:
            out = self.summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )
            summaries.append(out[0]["summary_text"])
        return " ".join(summaries)

    @staticmethod
    def _chunk(text: str, max_tokens: int = 900) -> list[str]:
        words = text.split()
        return [
            " ".join(words[i : i + max_tokens])
            for i in range(0, len(words), max_tokens)
        ]


# ─────────────────────────────────────────────
# 3. SENTIMENT ANALYSIS (bonus)
# ─────────────────────────────────────────────
class SentimentAnalyzer:
    def __init__(self):
        self.pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

    def analyze(self, text: str) -> dict:
        result = self.pipe(text[:512])[0]
        return {"label": result["label"], "confidence": round(result["score"], 4)}


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
SAMPLE_TEXT = """
Artificial intelligence (AI) is transforming every sector of the global economy.
From healthcare to finance, AI-powered systems are helping professionals make
faster and more accurate decisions. In medicine, deep learning models can now
detect cancers in radiology images with accuracy that rivals seasoned doctors.
In finance, algorithmic trading systems powered by reinforcement learning
generate millions of micro-decisions per second. Natural language processing
has enabled virtual assistants that understand context and nuance, while
computer vision enables self-driving cars to navigate complex urban environments.
Despite these advances, significant challenges remain. Ethical concerns around
bias, fairness, and accountability are at the forefront of AI policy discussions
worldwide. Researchers are actively working on explainability techniques so that
humans can understand and trust AI decisions. The next decade promises even
greater breakthroughs, with multimodal models that can reason across text,
images, audio, and video simultaneously.
"""


def main():
    parser = argparse.ArgumentParser(description="NLP Text Summarization Demo")
    parser.add_argument(
        "--mode",
        choices=["extractive", "abstractive", "sentiment", "all"],
        default="all",
    )
    parser.add_argument("--text", type=str, default=SAMPLE_TEXT)
    parser.add_argument("--sentences", type=int, default=3)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  NLP TEXT SUMMARIZER")
    print("=" * 60)
    print("\n📄 Original Text:")
    print(textwrap.fill(args.text.strip(), 70))

    if args.mode in ("extractive", "all"):
        print("\n─── Extractive Summary ──────────────────────────────────")
        ext = ExtractiveSummarizer()
        print(textwrap.fill(ext.summarize(args.text, args.sentences), 70))

    if args.mode in ("abstractive", "all"):
        print("\n─── Abstractive Summary (BART) ──────────────────────────")
        abst = AbstractiveSummarizer()
        print(textwrap.fill(abst.summarize(args.text), 70))

    if args.mode in ("sentiment", "all"):
        print("\n─── Sentiment Analysis ──────────────────────────────────")
        sa = SentimentAnalyzer()
        result = sa.analyze(args.text)
        print(f"  Label      : {result['label']}")
        print(f"  Confidence : {result['confidence']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
