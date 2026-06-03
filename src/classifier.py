"""
Car Type Classifier — Familiale vs Sportive
Utilisation d'un LLM pré-entraîné en zero-shot (facebook/bart-large-mnli)

Usage :
    python src/classifier.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from transformers import pipeline

# ─────────────────────────────────────────
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
# ─────────────────────────────────────────

def load_and_clean_data(data_dir: str = "data/") -> pd.DataFrame:
    """Charge et nettoie les datasets CSV depuis le dossier data/."""
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        print("⚠️  Aucun CSV trouvé dans data/. Génération d'un dataset de démonstration...")
        return generate_demo_dataset()

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(os.path.join(data_dir, f), encoding="utf-8", on_bad_lines="skip")
            dfs.append(df)
            print(f"✅ Chargé : {f} ({len(df)} lignes, {df.shape[1]} colonnes)")
        except Exception as e:
            print(f"❌ Erreur lecture {f} : {e}")

    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates()
    df = df.dropna(subset=[c for c in df.columns if "name" in c or "car" in c], how="all")
    print(f"\n📊 Dataset final : {len(df)} lignes, {df.shape[1]} colonnes")
    return df


def generate_demo_dataset() -> pd.DataFrame:
    """
    Dataset de démonstration si aucun CSV n'est disponible.
    Contient 40 voitures avec leurs caractéristiques.
    """
    data = [
        # Familiales
        ("Renault Espace", 7, 140, 200, "Diesel", "family"),
        ("Volkswagen Touran", 7, 150, 195, "Diesel", "family"),
        ("Ford S-Max", 7, 160, 205, "Petrol", "family"),
        ("Toyota Verso", 7, 130, 185, "Hybrid", "family"),
        ("Citroën C4 Picasso", 7, 120, 190, "Diesel", "family"),
        ("Peugeot 5008", 7, 130, 198, "Diesel", "family"),
        ("Seat Alhambra", 7, 150, 200, "Diesel", "family"),
        ("Skoda Octavia Combi", 5, 150, 210, "Petrol", "family"),
        ("Dacia Lodgy", 7, 115, 175, "LPG", "family"),
        ("Opel Zafira", 7, 140, 195, "Petrol", "family"),
        ("BMW X5", 5, 265, 240, "Diesel", "family"),
        ("Volvo XC90", 7, 235, 210, "Hybrid", "family"),
        ("Honda CR-V", 5, 193, 192, "Hybrid", "family"),
        ("Kia Sorento", 7, 200, 200, "Diesel", "family"),
        ("Hyundai Santa Fe", 7, 200, 200, "Diesel", "family"),
        ("Toyota Land Cruiser", 7, 204, 210, "Diesel", "family"),
        ("Nissan Qashqai", 5, 140, 186, "Petrol", "family"),
        ("Renault Scenic", 5, 130, 195, "Diesel", "family"),
        ("Ford Galaxy", 7, 150, 205, "Diesel", "family"),
        ("Volkswagen Sharan", 7, 150, 200, "Diesel", "family"),
        # Sportives
        ("Ferrari 488 GTB", 2, 670, 330, "Petrol", "sport"),
        ("Porsche 911 Carrera", 2, 450, 293, "Petrol", "sport"),
        ("Lamborghini Huracán", 2, 610, 325, "Petrol", "sport"),
        ("BMW M3", 4, 510, 290, "Petrol", "sport"),
        ("Mercedes-AMG GT", 2, 476, 310, "Petrol", "sport"),
        ("Audi R8", 2, 570, 320, "Petrol", "sport"),
        ("Chevrolet Corvette C8", 2, 495, 312, "Petrol", "sport"),
        ("Dodge Challenger SRT", 4, 707, 317, "Petrol", "sport"),
        ("Ford Mustang GT500", 4, 760, 290, "Petrol", "sport"),
        ("McLaren 720S", 2, 720, 341, "Petrol", "sport"),
        ("Jaguar F-Type R", 2, 575, 300, "Petrol", "sport"),
        ("Alfa Romeo Giulia QV", 4, 510, 307, "Petrol", "sport"),
        ("Renault Mégane RS", 4, 300, 260, "Petrol", "sport"),
        ("Honda Civic Type R", 4, 320, 272, "Petrol", "sport"),
        ("Toyota GR Yaris", 4, 261, 230, "Petrol", "sport"),
        ("Volkswagen Golf R", 4, 320, 270, "Petrol", "sport"),
        ("Subaru BRZ", 4, 234, 226, "Petrol", "sport"),
        ("Mazda MX-5", 2, 184, 219, "Petrol", "sport"),
        ("BMW Z4 M40i", 2, 340, 250, "Petrol", "sport"),
        ("Aston Martin Vantage", 2, 510, 314, "Petrol", "sport"),
    ]
    df = pd.DataFrame(data, columns=["car_name", "seats", "hp", "speed_kmh", "fuel", "true_label"])
    print(f"✅ Dataset de démonstration généré : {len(df)} voitures")
    return df


# ─────────────────────────────────────────
# 2. CONSTRUCTION DU TEXTE D'ENTRÉE
# ─────────────────────────────────────────

def build_text_description(row: pd.Series) -> str:
    """
    Construit une description textuelle d'une voiture à partir de ses attributs.
    Compatible avec les colonnes des datasets Kaggle et le dataset de démonstration.
    """
    parts = []

    # Nom du véhicule
    for col in ["car_name", "car name", "name", "model", "brand"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
            break

    # Nombre de places
    for col in ["seats", "seating_capacity", "passengers", "capacity"]:
        if col in row.index and pd.notna(row[col]):
            try:
                seats = int(float(row[col]))
                parts.append(f"{seats} seats")
            except:
                pass
            break

    # Puissance
    for col in ["hp", "horsepower", "power_hp", "engine_hp"]:
        if col in row.index and pd.notna(row[col]):
            try:
                hp = int(float(str(row[col]).replace("hp", "").strip()))
                parts.append(f"{hp} HP")
            except:
                pass
            break

    # Vitesse max
    for col in ["speed_kmh", "top_speed", "max_speed", "speed"]:
        if col in row.index and pd.notna(row[col]):
            try:
                speed = int(float(str(row[col]).replace("km/h", "").strip()))
                parts.append(f"top speed {speed} km/h")
            except:
                pass
            break

    # Carburant
    for col in ["fuel", "fuel_type", "energy"]:
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
            break

    return ", ".join(parts) if parts else "unknown car"


# ─────────────────────────────────────────
# 3. CLASSIFICATION ZERO-SHOT
# ─────────────────────────────────────────

def classify_cars(df: pd.DataFrame, text_col: str = "text_description") -> pd.DataFrame:
    """Applique la classification zero-shot sur la colonne texte."""
    print("\n🤖 Chargement du modèle facebook/bart-large-mnli...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1  # CPU ; mettre 0 pour GPU si disponible
    )

    labels = ["family car", "sport car"]
    predictions = []
    scores_family = []
    scores_sport = []

    print(f"🔍 Classification de {len(df)} voitures...")
    for text in tqdm(df[text_col], desc="Classifying"):
        result = classifier(text, candidate_labels=labels)
        pred = result["labels"][0]  # Label avec le score le plus élevé
        predictions.append(pred)
        # Stocker les scores pour analyse
        score_dict = dict(zip(result["labels"], result["scores"]))
        scores_family.append(score_dict.get("family car", 0))
        scores_sport.append(score_dict.get("sport car", 0))

    df["predicted_label"] = predictions
    df["score_family"] = scores_family
    df["score_sport"] = scores_sport
    df["predicted_class"] = df["predicted_label"].map({"family car": "family", "sport car": "sport"})
    return df


# ─────────────────────────────────────────
# 4. ÉVALUATION
# ─────────────────────────────────────────

def evaluate(df: pd.DataFrame, true_col: str = "true_label", pred_col: str = "predicted_class"):
    """Calcule et affiche les métriques d'évaluation."""
    y_true = df[true_col]
    y_pred = df[pred_col]

    print("\n" + "="*50)
    print("📊 RÉSULTATS D'ÉVALUATION")
    print("="*50)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="sport", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="sport", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="sport", zero_division=0)

    print(f"  Accuracy  : {acc:.2%}")
    print(f"  Precision : {prec:.2%}")
    print(f"  Recall    : {rec:.2%}")
    print(f"  F1-score  : {f1:.2%}")
    print("\nRapport détaillé :")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ─────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────

def plot_results(df: pd.DataFrame, metrics: dict, output_dir: str = "results/"):
    """Génère et sauvegarde les graphiques d'évaluation."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- Matrice de confusion ---
    if "true_label" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        cm = confusion_matrix(df["true_label"], df["predicted_class"], labels=["family", "sport"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Familiale", "Sportive"],
                    yticklabels=["Familiale", "Sportive"], ax=axes[0])
        axes[0].set_title("Matrice de confusion", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Prédiction")
        axes[0].set_ylabel("Réalité")

        # --- Métriques bar chart ---
        metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
        metric_values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
        bars = axes[1].bar(metric_names, metric_values, color=colors, edgecolor="white", linewidth=1.5)
        axes[1].set_ylim(0, 1.1)
        axes[1].set_title("Métriques d'évaluation", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Score")
        for bar, val in zip(bars, metric_values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f"{val:.2%}", ha="center", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "evaluation_metrics.png"), dpi=150)
        print(f"\n✅ Graphique sauvegardé : {output_dir}evaluation_metrics.png")
        plt.show()

    # --- Distribution des scores de confiance ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["score_family"], bins=20, alpha=0.7, label="Score Familiale", color="#4C72B0")
    ax.hist(df["score_sport"], bins=20, alpha=0.7, label="Score Sportive", color="#C44E52")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Seuil 0.5")
    ax.set_title("Distribution des scores de confiance (zero-shot)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Score de confiance")
    ax.set_ylabel("Nombre de voitures")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_scores.png"), dpi=150)
    print(f"✅ Graphique sauvegardé : {output_dir}confidence_scores.png")
    plt.show()


# ─────────────────────────────────────────
# 6. PIPELINE PRINCIPAL
# ─────────────────────────────────────────

def main():
    print("🚗 === Car Type Classifier — LLM Zero-Shot ===\n")

    # Chargement
    df = load_and_clean_data("data/")

    # Construction des descriptions textuelles
    print("\n📝 Construction des descriptions textuelles...")
    df["text_description"] = df.apply(build_text_description, axis=1)
    print("Exemples de descriptions :")
    for txt in df["text_description"].head(3):
        print(f"  → {txt}")

    # Classification
    df = classify_cars(df)

    # Affichage d'un échantillon
    print("\n🔎 Exemples de prédictions :")
    cols_to_show = ["text_description", "predicted_label", "score_family", "score_sport"]
    if "true_label" in df.columns:
        cols_to_show.insert(1, "true_label")
    print(df[cols_to_show].head(10).to_string(index=False))

    # Évaluation (seulement si les vraies étiquettes sont disponibles)
    metrics = {}
    if "true_label" in df.columns:
        metrics = evaluate(df)
    else:
        print("\n⚠️  Pas de colonne 'true_label' trouvée : évaluation ignorée.")
        print("   Pour évaluer, ajoutez une colonne 'true_label' avec les valeurs 'family' ou 'sport'.")

    # Sauvegarde des résultats
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/predictions.csv", index=False)
    print("\n✅ Prédictions sauvegardées : results/predictions.csv")

    # Visualisations
    plot_results(df, metrics)

    print("\n🎉 Pipeline terminé avec succès !")


if __name__ == "__main__":
    main()
