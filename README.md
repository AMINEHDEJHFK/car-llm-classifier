# 🚗 Car Type Classifier — Familiale vs Sportive

> Projet réalisé dans le cadre du cours **Large Language Models** — Échéance : 8 juin 2026

## 📌 Problématique

Peut-on automatiquement classifier une voiture comme **familiale** ou **sportive** à partir de sa description textuelle (nombre de places, puissance, type de carrosserie, usage) grâce à un LLM pré-entraîné en **zero-shot** ?

---

## 🗂️ Structure du projet

```
car-llm-classifier/
├── data/                        ← Datasets CSV (à télécharger, voir section Données)
├── notebooks/
│   ├── 01_exploration.ipynb     ← Exploration et visualisation des données
│   ├── 02_preprocessing.ipynb   ← Nettoyage et construction du texte
│   └── 03_classification.ipynb  ← Classification zero-shot + évaluation
├── src/
│   └── classifier.py            ← Script Python autonome (pipeline complet)
├── results/                     ← Graphiques et métriques générés
├── requirements.txt
└── README.md
```

---

## 📦 Données

Deux datasets Kaggle combinés :

| Dataset | Lien |
|---|---|
| Cars Datasets 2025 | https://www.kaggle.com/datasets/abdulmalik1518/cars-datasets-2025 |
| Large Dataset of Cars | https://www.kaggle.com/datasets/makslypko/large-cars-dataset |

**Téléchargement via Kaggle CLI :**
```bash
pip install kaggle
kaggle datasets download -d abdulmalik1518/cars-datasets-2025 -p data/
kaggle datasets download -d makslypko/large-cars-dataset -p data/
cd data && unzip "*.zip"
```

> 💡 Vous aurez besoin d'un compte Kaggle et d'un fichier `~/.kaggle/kaggle.json`.

---

## ⚙️ Installation

```bash
git clone https://github.com/VOTRE_USERNAME/car-llm-classifier.git
cd car-llm-classifier
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Option 1 — Script Python autonome

```bash
python src/classifier.py
```

Génère automatiquement les métriques et les graphiques dans `results/`.

### Option 2 — Notebooks Jupyter (recommandé)

```bash
jupyter notebook
```

Ouvrir les notebooks dans l'ordre :
1. `01_exploration.ipynb`
2. `02_preprocessing.ipynb`
3. `03_classification.ipynb`

---

## 🔬 Approche technique

- **Modèle** : `facebook/bart-large-mnli` (zero-shot classification, HuggingFace)
- **Labels** : `["family car", "sport car"]`
- **Input** : Texte construit à partir des colonnes CSV (nom, places, puissance, carburant, vitesse)
- **Évaluation** : Accuracy, Precision, Recall, F1-score, Matrice de confusion

---

## 📊 Résultats attendus

| Métrique | Description |
|---|---|
| Accuracy | % de voitures correctement classifiées |
| F1-score | Équilibre précision/rappel |
| Matrice de confusion | Visualisation des erreurs |

---

## 🧠 Modèle utilisé

`facebook/bart-large-mnli` est un modèle de 400M paramètres fine-tuné sur MNLI (Multi-Genre Natural Language Inference). Il permet la classification zero-shot sans aucun entraînement supplémentaire.

---

## 👤 Auteur

Projet réalisé dans le cadre du Master IA — Cours Large Language Models 2025/2026.
