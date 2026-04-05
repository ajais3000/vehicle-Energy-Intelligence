# =============================================================================
# FEATURE 3: Driver Behavior Fingerprinting & Real-Time Eco-Score
# FILE: features/feature3_driver_behavior/ml_model.py
#
# PURPOSE:
#   Trains a Random Forest Classifier on the KMeans-labelled driving style data
#   to predict: Eco (0), Moderate (1), Aggressive (2).
#   The predict_proba() output is converted to an Eco-Score (0–100).
#
# WHY RANDOM FOREST CLASSIFIER?
#   1. RF handles the mixed features well (speed, RPM, energy consumption)
#      without requiring strict distributional assumptions.
#   2. RF with class_weight='balanced' naturally handles unequal cluster sizes
#      (Moderate typically dominates; Eco and Aggressive are smaller groups).
#   3. predict_proba() gives soft probabilities needed for the Eco-Score formula:
#      Eco-Score = P(Eco)x100 + P(Moderate)x50 + P(Aggressive)x0
#   4. Feature importance from RF confirms which behaviours truly drive the
#      classification — a transparency requirement for driver coaching.
# =============================================================================

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def train_model():
    """Train RF Classifier for driving style. Saves model to MODEL_DIR."""
    X_train = joblib.load(os.path.join(MODEL_DIR, "f3_X_train.pkl"))
    X_test  = joblib.load(os.path.join(MODEL_DIR, "f3_X_test.pkl"))
    y_train = joblib.load(os.path.join(MODEL_DIR, "f3_y_train.pkl"))
    y_test  = joblib.load(os.path.join(MODEL_DIR, "f3_y_test.pkl"))

    print(f"[F3] Training on {X_train.shape[0]:,} samples, {X_train.shape[1]} features")

    # class_weight='balanced': Adjusts sample weights inversely proportional
    # to class frequency. Prevents the model from defaulting to 'Moderate'
    # (the majority class) and ignoring Eco/Aggressive minority classes.
    # n_estimators=200: See HVAC reasoning. 200 trees for robust probability estimates.
    # max_depth=10: Slightly shallower than regression RF because classification
    # boundaries are simpler; deeper trees would overfit KMeans cluster noise.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[F3] Accuracy: {acc:.4f}")
    print("[F3] Classification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=["Eco", "Moderate", "Aggressive"]))

    features = joblib.load(os.path.join(MODEL_DIR, "f3_features.pkl"))
    imp = dict(zip(features, model.feature_importances_))
    print("[F3] Feature importances:", {k: f"{v:.3f}" for k, v in
          sorted(imp.items(), key=lambda x: -x[1])})

    joblib.dump(model, os.path.join(MODEL_DIR, "f3_behavior_model.pkl"))
    print("[F3] Model saved -> models/f3_behavior_model.pkl")

    return {"model": model, "accuracy": acc}


if __name__ == "__main__":
    result = train_model()
    print(f"[F3] Training complete. Accuracy={result['accuracy']:.4f}")
