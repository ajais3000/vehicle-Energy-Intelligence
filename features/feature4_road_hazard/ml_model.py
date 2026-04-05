# =============================================================================
# FEATURE 4: Road Hazard Risk Predictor
# FILE: features/feature4_road_hazard/ml_model.py
#
# PURPOSE:
#   Trains a Random Forest Classifier to predict road hazard risk level:
#   Low (0), Medium (1), High (2) from OBD + road metadata.
#
# WHY RANDOM FOREST CLASSIFIER?
#   1. Hazard risk labelling is inherently imbalanced (most driving is safe).
#      RF with class_weight='balanced' reweights samples to prevent the model
#      from always predicting 'Low' to achieve high accuracy.
#   2. RF handles the mix of binary flags (Intersection, Ice_Risk),
#      continuous values (Speed, RPM), and engineered interactions
#      (Speed_Excess) naturally without requiing normalisation.
#   3. High interpretability: feature importance reveals which signals the
#      model uses — crucial for a safety-critical prediction system.
#   4. predict_proba() supports confidence thresholds: alerts only fire
#      when P(High) > 0.6, reducing false alarms for driver trust.
# =============================================================================

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def train_model():
    """Train RF Classifier for hazard risk. Saves model to MODEL_DIR."""
    X_train = joblib.load(os.path.join(MODEL_DIR, "f4_X_train.pkl"))
    X_test  = joblib.load(os.path.join(MODEL_DIR, "f4_X_test.pkl"))
    y_train = joblib.load(os.path.join(MODEL_DIR, "f4_y_train.pkl"))
    y_test  = joblib.load(os.path.join(MODEL_DIR, "f4_y_test.pkl"))

    print(f"[F4] Training on {X_train.shape[0]:,} samples, {X_train.shape[1]} features")

    # n_estimators=300: Safety-critical prediction benefits from more trees
    #   (lower variance in probability estimates -> more reliable alert thresholds).
    # max_depth=15: Hazard prediction requires capturing complex multi-factor
    #   interactions (speed excess + intersection + icy road + high load).
    # class_weight='balanced': Corrects for Low-risk dominance. Without this,
    #   the model would learn to always predict 'Low' and achieve ~70% accuracy
    #   but completely miss High risk events — the worst possible outcome.
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[F4] Accuracy: {acc:.4f}")
    print("[F4] Classification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=["Low", "Medium", "High"]))

    features = joblib.load(os.path.join(MODEL_DIR, "f4_features.pkl"))
    imp = dict(zip(features, model.feature_importances_))
    print("[F4] Feature importances:", {k: f"{v:.3f}" for k, v in
          sorted(imp.items(), key=lambda x: -x[1])})

    joblib.dump(model, os.path.join(MODEL_DIR, "f4_hazard_model.pkl"))
    print("[F4] Model saved -> models/f4_hazard_model.pkl")

    return {"model": model, "accuracy": acc}


if __name__ == "__main__":
    result = train_model()
    print(f"[F4] Training complete. Accuracy={result['accuracy']:.4f}")
