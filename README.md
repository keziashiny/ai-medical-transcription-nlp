# Medical NLP Assistant (Semantic Search + Chat)

Live demo: https://ai-medical-transcription-nlp-k8mdipwez62jhusabtxri8.streamlit.app/

A Streamlit web app that lets you explore a medical transcription dataset using semantic search and a chat-style assistant. Built for educational/demo use (not medical advice).

## What you can do
- 🔍 **Search Mode:** type a symptom/condition and get the most similar clinical notes
- 💬 **Chat Mode:** ask a question and get a human-style summary grounded in retrieved notes
- 🧾 Optional: view cleaned model input (hidden behind an expander)

## How it works (high level)
1. Text is embedded using a SentenceTransformer model.
2. A NearestNeighbors index retrieves top-k similar notes for a query.
3. The UI displays the matched notes and a friendly summary.

## Tech stack
- Python, Streamlit
- sentence-transformers
- scikit-learn (NearestNeighbors)
- pandas, numpy

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
