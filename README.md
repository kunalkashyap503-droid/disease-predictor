# 🧬 MedScan — AI Disease Prediction System

A full Python ML project that predicts diseases from symptoms using a
Random Forest classifier, with both a Flask web UI and a CLI.

## Features
- **15 diseases** predicted from **30 symptoms**
- **Random Forest** classifier (95%+ accuracy on test set)
- **Flask web app** with interactive symptom selector
- **CLI tool** for terminal-based predictions
- Top-3 predictions with confidence scores
- Precautions & medicine recommendations per disease

## Diseases Covered
Common Cold, Influenza, COVID-19, Pneumonia, Bronchitis,
Malaria, Dengue Fever, Typhoid, Tuberculosis, Hepatitis,
Diabetes, Migraine, Gastroenteritis, UTI, Allergies

## Setup

```bash
pip install -r requirements.txt
```

## Run Web App

```bash
python app.py
```
Open http://localhost:5000

## Run CLI

```bash
python cli.py
```

## Train Model Only

```bash
python model.py
```

## Project Structure

```
disease_prediction/
├── model.py          # ML model, training, prediction logic
├── app.py            # Flask web application
├── cli.py            # Terminal-based predictor
├── requirements.txt  # Dependencies
└── models/           # Saved model (auto-created)
    └── disease_model.pkl
```

## ⚠️ Disclaimer
For educational purposes only. Not a substitute for professional medical advice.
