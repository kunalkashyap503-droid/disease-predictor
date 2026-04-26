"""
Disease Prediction Model
Trains a Random Forest classifier on symptom data to predict diseases.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── Symptom & Disease Data ──────────────────────────────────────────────────

SYMPTOMS = [
    "fever", "cough", "fatigue", "difficulty_breathing", "headache",
    "sore_throat", "runny_nose", "body_ache", "nausea", "vomiting",
    "diarrhea", "chest_pain", "dizziness", "loss_of_appetite",
    "joint_pain", "skin_rash", "sweating", "chills", "abdominal_pain",
    "back_pain", "loss_of_taste", "loss_of_smell", "sneezing",
    "muscle_weakness", "swollen_lymph_nodes", "yellow_skin",
    "dark_urine", "frequent_urination", "blurred_vision", "numbness"
]

DISEASES = [
    "Common Cold", "Influenza", "Pneumonia", "Bronchitis",
    "Malaria", "Dengue Fever", "Typhoid", "Tuberculosis", "Hepatitis",
    "Diabetes", "Migraine", "Gastroenteritis", "Urinary Tract Infection",
    "Allergies"
]

# Symptom profiles: each disease maps to its typical symptoms
DISEASE_PROFILES = {
    "Common Cold":              ["runny_nose", "sore_throat", "sneezing", "cough", "headache", "fatigue"],
    "Influenza":                ["fever", "cough", "body_ache", "fatigue", "headache", "chills", "sweating"],
    "Pneumonia":                ["fever", "cough", "difficulty_breathing", "chest_pain", "fatigue", "chills"],
    "Bronchitis":               ["cough", "fatigue", "chest_pain", "difficulty_breathing", "sore_throat"],
    "Malaria":                  ["fever", "chills", "sweating", "headache", "nausea", "vomiting", "fatigue", "body_ache"],
    "Dengue Fever":             ["fever", "headache", "joint_pain", "body_ache", "skin_rash", "nausea", "fatigue"],
    "Typhoid":                  ["fever", "headache", "abdominal_pain", "loss_of_appetite", "nausea", "fatigue"],
    "Tuberculosis":             ["cough", "fatigue", "fever", "sweating", "loss_of_appetite", "chest_pain"],
    "Hepatitis":                ["yellow_skin", "dark_urine", "fatigue", "nausea", "abdominal_pain", "loss_of_appetite"],
    "Diabetes":                 ["frequent_urination", "fatigue", "blurred_vision", "loss_of_appetite", "numbness"],
    "Migraine":                 ["headache", "nausea", "blurred_vision", "dizziness", "fatigue"],
    "Gastroenteritis":          ["nausea", "vomiting", "diarrhea", "abdominal_pain", "fever", "fatigue"],
    "Urinary Tract Infection":  ["frequent_urination", "abdominal_pain", "fatigue", "fever", "back_pain"],
    "Allergies":                ["runny_nose", "sneezing", "skin_rash", "cough", "headache", "blurred_vision"],
}

# Precautions & medicines per disease
DISEASE_INFO = {
    "Common Cold": {
        "description": "A viral infection of the upper respiratory tract.",
        "precautions": ["Rest well", "Stay hydrated", "Avoid cold drinks", "Wash hands frequently"],
        "medicines": ["Paracetamol", "Antihistamines", "Decongestants", "Vitamin C"],
        "severity": "Mild",
    },
    "Influenza": {
        "description": "A contagious respiratory illness caused by influenza viruses.",
        "precautions": ["Rest at home", "Stay hydrated", "Avoid contact with others", "Get vaccinated"],
        "medicines": ["Oseltamivir (Tamiflu)", "Paracetamol", "Ibuprofen", "Cough syrup"],
        "severity": "Moderate",
    },
    "Pneumonia": {
        "description": "Infection that inflames air sacs in one or both lungs.",
        "precautions": ["Seek immediate medical attention", "Rest", "Stay warm", "Avoid smoking"],
        "medicines": ["Antibiotics", "Paracetamol", "Cough suppressants", "Oxygen therapy"],
        "severity": "High",
    },
    "Bronchitis": {
        "description": "Inflammation of the lining of bronchial tubes.",
        "precautions": ["Avoid smoking", "Use humidifier", "Rest", "Drink warm liquids"],
        "medicines": ["Bronchodilators", "Cough suppressants", "Paracetamol", "Antibiotics (if bacterial)"],
        "severity": "Moderate",
    },
    "Malaria": {
        "description": "A life-threatening disease caused by Plasmodium parasites transmitted by mosquitoes.",
        "precautions": ["Use mosquito nets", "Apply insect repellent", "Seek medical care", "Take antimalarials"],
        "medicines": ["Chloroquine", "Artemisinin combination therapy", "Primaquine", "Paracetamol"],
        "severity": "High",
    },
    "Dengue Fever": {
        "description": "A mosquito-borne viral infection causing severe flu-like illness.",
        "precautions": ["Avoid mosquito bites", "Stay hydrated", "Monitor platelet count", "Hospitalize if severe"],
        "medicines": ["Paracetamol", "IV fluids", "Platelet transfusion (severe)", "Rest"],
        "severity": "High",
    },
    "Typhoid": {
        "description": "A bacterial infection caused by Salmonella typhi, spread through contaminated food/water.",
        "precautions": ["Drink clean water", "Eat hygienic food", "Wash hands", "Get vaccinated"],
        "medicines": ["Ciprofloxacin", "Azithromycin", "Ceftriaxone", "Paracetamol"],
        "severity": "Moderate-High",
    },
    "Tuberculosis": {
        "description": "A potentially serious infectious disease mainly affecting the lungs.",
        "precautions": ["Complete full medication course", "Wear mask", "Ventilate rooms", "Regular checkups"],
        "medicines": ["Isoniazid", "Rifampicin", "Ethambutol", "Pyrazinamide"],
        "severity": "High",
    },
    "Hepatitis": {
        "description": "Inflammation of the liver, commonly caused by a viral infection.",
        "precautions": ["Avoid alcohol", "Eat healthy diet", "Get vaccinated", "Avoid sharing needles"],
        "medicines": ["Antiviral drugs", "Interferon", "Liver supplements", "Rest"],
        "severity": "Moderate-High",
    },
    "Diabetes": {
        "description": "A chronic condition that affects how the body processes blood sugar (glucose).",
        "precautions": ["Monitor blood sugar", "Follow diet plan", "Exercise regularly", "Regular checkups"],
        "medicines": ["Metformin", "Insulin", "Glipizide", "Blood sugar monitors"],
        "severity": "Chronic",
    },
    "Migraine": {
        "description": "A headache disorder characterized by recurrent attacks of severe headache.",
        "precautions": ["Avoid triggers", "Rest in dark quiet room", "Stay hydrated", "Manage stress"],
        "medicines": ["Sumatriptan", "Ibuprofen", "Paracetamol", "Anti-nausea medication"],
        "severity": "Moderate",
    },
    "Gastroenteritis": {
        "description": "Inflammation of the stomach and intestines, usually from infection.",
        "precautions": ["Stay hydrated (ORS)", "Eat bland foods", "Wash hands", "Avoid dairy"],
        "medicines": ["ORS (Oral Rehydration Solution)", "Loperamide", "Probiotics", "Antiemetics"],
        "severity": "Mild-Moderate",
    },
    "Urinary Tract Infection": {
        "description": "An infection in any part of the urinary system.",
        "precautions": ["Drink plenty of water", "Urinate frequently", "Maintain hygiene", "Avoid irritants"],
        "medicines": ["Trimethoprim", "Nitrofurantoin", "Ciprofloxacin", "Cranberry supplements"],
        "severity": "Moderate",
    },
    "Allergies": {
        "description": "An immune system reaction to a foreign substance not typically harmful.",
        "precautions": ["Identify and avoid triggers", "Keep windows closed", "Use air purifiers", "Carry antihistamines"],
        "medicines": ["Antihistamines", "Corticosteroids", "Decongestants", "Nasal sprays"],
        "severity": "Mild-Moderate",
    },
}

SEVERITY_COLOR = {
    "Mild": "#22c55e",
    "Mild-Moderate": "#84cc16",
    "Moderate": "#f59e0b",
    "Moderate-High": "#f97316",
    "High": "#ef4444",
    "Chronic": "#8b5cf6",
}


def generate_training_data(n_samples=3000, noise=0.15):
    """Generate synthetic training data based on disease symptom profiles."""
    np.random.seed(42)
    X, y = [], []

    samples_per_disease = n_samples // len(DISEASES)

    for disease in DISEASES:
        profile = DISEASE_PROFILES[disease]
        for _ in range(samples_per_disease):
            row = np.zeros(len(SYMPTOMS))
            # Core symptoms always present
            for symptom in profile:
                idx = SYMPTOMS.index(symptom)
                row[idx] = 1
            # Add noise: randomly flip some bits
            flip_mask = np.random.rand(len(SYMPTOMS)) < noise
            row = np.abs(row - flip_mask.astype(float))
            X.append(row)
            y.append(disease)

    return np.array(X), np.array(y)


def train_model():
    """Train and save the disease prediction model."""
    print("🔬 Generating training data...")
    X, y = generate_training_data(n_samples=3000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {acc * 100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    with open("models/disease_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("💾 Model saved to models/disease_model.pkl")
    return model, acc


def load_model():
    """Load the saved model, training if needed."""
    model_path = "models/disease_model.pkl"
    if not os.path.exists(model_path):
        print("No saved model found. Training new model...")
        model, _ = train_model()
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    return model


def predict_disease(symptom_list, model):
    """
    Predict disease from a list of symptom names.
    Returns top-3 predictions with probabilities.
    """
    input_vec = np.zeros(len(SYMPTOMS))
    for s in symptom_list:
        s = s.strip().lower().replace(" ", "_")
        if s in SYMPTOMS:
            input_vec[SYMPTOMS.index(s)] = 1

    proba = model.predict_proba([input_vec])[0]
    classes = model.classes_

    top_indices = np.argsort(proba)[::-1][:3]
    results = []
    for i in top_indices:
        disease = classes[i]
        confidence = round(float(proba[i]) * 100, 1)
        info = DISEASE_INFO.get(disease, {})
        results.append({
            "disease": disease,
            "confidence": confidence,
            "description": info.get("description", ""),
            "precautions": info.get("precautions", []),
            "medicines": info.get("medicines", []),
            "severity": info.get("severity", "Unknown"),
            "severity_color": SEVERITY_COLOR.get(info.get("severity", ""), "#6b7280"),
        })
    return results


if __name__ == "__main__":
    train_model()
