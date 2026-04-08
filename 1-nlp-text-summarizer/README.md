# 📝 NLP Text Summarizer

A production-ready **Natural Language Processing** pipeline that performs:
- ✅ **Extractive Summarization** — TF-IDF + NLTK sentence scoring
- ✅ **Abstractive Summarization** — Facebook BART (`facebook/bart-large-cnn`)
- ✅ **Sentiment Analysis** — DistilBERT SST-2

## 🏗️ Architecture

```
Input Text
    │
    ├──► Extractive Engine (NLTK FreqDist) ──► Top-N Sentences
    │
    ├──► Abstractive Engine (BART Seq2Seq) ──► Generated Summary
    │
    └──► Sentiment Classifier (DistilBERT)  ──► Label + Confidence
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all modes
python summarizer.py --mode all

# Abstractive only
python summarizer.py --mode abstractive

# Custom text
python summarizer.py --mode extractive --text "Your long text here..." --sentences 2
```

## 📊 Sample Output

```
─── Extractive Summary ───────────────────────────────
Artificial intelligence (AI) is transforming every sector of the
global economy. In medicine, deep learning models can now detect
cancers in radiology images with accuracy that rivals seasoned doctors.

─── Abstractive Summary (BART) ───────────────────────
AI is transforming healthcare, finance, and other industries. Despite
advances, ethical concerns remain around bias and fairness.

─── Sentiment Analysis ───────────────────────────────
  Label      : POSITIVE
  Confidence : 0.9987
```

## 🧠 Models Used

| Task | Model | Params |
|------|-------|--------|
| Abstractive Summarization | `facebook/bart-large-cnn` | 400M |
| Sentiment Analysis | `distilbert-base-uncased-finetuned-sst-2-english` | 66M |

## 📁 Project Structure

```
1-nlp-text-summarizer/
├── summarizer.py      # Main pipeline
├── requirements.txt
└── README.md
```

## 🤝 Contributing
Pull requests are welcome!

## 📄 License
MIT
