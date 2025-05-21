# 🧠 Word Embedding Models: Concept & Implementation

This folder contains my study notes and clean PyTorch implementations of **word embedding models**, especially:
- `CBOW` (Continuous Bag of Words)
- `Skip-gram`

---

## 📌 What Are Embeddings?

In NLP, **embeddings** are dense vector representations of words or phrases. Instead of using one-hot encoding (which is sparse and meaningless), embeddings **capture semantic relationships** between words — e.g., "heart" and "cardiology" should be closer in vector space than "heart" and "car".

> Embeddings are the foundational layer in almost all modern LLMs.

---

## 🧬 Embedding in LLMs vs Word2Vec

- In **LLMs** (e.g., GPT, BERT), the embedding layer is **learned jointly with the entire model**, and includes **positional encoding**.
- In **Word2Vec**, embeddings are trained **independently** using a shallow model and a local prediction task (CBOW or Skip-gram).

| Model      | Context window? | Output | Trains jointly? |
|------------|------------------|--------|------------------|
| Word2Vec   | Yes (small)       | Word   | No (frozen after) |
| LLMs (GPT) | Yes (global, via attention) | Tokens | Yes |

---

## 🧰 CBOW vs Skip-gram

### CBOW (Continuous Bag of Words)
- Goal: **Predict the center word** from surrounding context
- Efficient for frequent words

### Skip-gram
- Goal: **Predict context words** from a center word
- Works better for rare words

Both use a shallow neural net with one hidden layer (embedding layer).

---

## 🔁 Other Embedding Models (Beyond Word2Vec)

| Model      | Description |
|------------|-------------|
| **GloVe**      | Global word co-occurrence matrix + factorization (non-neural) |
| **FastText**  | Adds subword (n-gram) embeddings for better OOV handling |
| **ELMo**      | Contextualized embeddings from BiLSTM |
| **BERT embeddings** | Context-sensitive, layer-specific vectors from Transformers |

You can experiment with these using the `transformers` or `gensim` libraries.

---

## 🏥 How I Use Embeddings in My Medical AI Projects

These models can be used in many places within my [`ai-medical-projects`](https://github.com/yourname/ai-medical-projects) repo:

- 🧠 `surgery_monitoring/`  
  Embed patient-written feedback and compare with recovery-related expert texts

- 🩺 `outpatient_bot/`  
  Convert patient utterances into vectors to classify intents or recommend triage actions

- 🫀 `tavi_selector/`  
  Transform EHR-like structured text (e.g., comorbidities) into feature vectors for downstream rules or ML classifiers

---

## 📦 Folder Structure

```bash
embedding/
├── word2vec_cbow.py     # Predict center word from context
├── skipgram.py          # Predict context words from center
└── README.md            # You're reading it!
