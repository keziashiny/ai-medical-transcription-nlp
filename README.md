# AI Medical Transcription NLP

This project uses real medical transcription text (mtsamples dataset) and NLP (spaCy) to extract useful information from clinical-style notes.

## Current Features

- Loads medical transcription data from CSV (`mtsamples.csv`)
- Runs a baseline spaCy NER model on sample transcriptions
- Prints entities like dates, locations, possible medications, and other key terms

## Project Structure

- `data/raw/mtsamples.csv` — medical transcription dataset from Kaggle
- `src/data_loader.py` — loads and cleans the dataset
- `src/ner_baseline.py` — runs spaCy NER on one sample transcription
- `notebooks/` — will contain Jupyter notebooks for EDA and experiments

## How to Run

From the project folder:

```bash
python -m src.data_loader
python -m src.ner_baseline
