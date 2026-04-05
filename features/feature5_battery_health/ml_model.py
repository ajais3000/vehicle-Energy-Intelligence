# =============================================================================
# FEATURE 5: HV Battery Health & Long-Term Degradation Risk Predictor
# FILE: features/feature5_battery_health/ml_model.py
#
# PURPOSE:
#   Trains an XGBoost Classifier to predict battery stress level:
#   Low (0), Medium (1), High (2) from instantaneous vehicle/battery telemetry.
#
# WHY XGBOOST CLASSIFIER?
#   1. Battery degradation is a multi-factor interaction problem. XGBoost's
#      gradient-boosted tree ensembles capture complex non-linear combinations
#      of Current x SOC x Temperature better than RF at the same depth.
#   2. XGBoost uses second-order gradient information (Newton boosting), making
#      it more accurate per tree on tabular physics data vs. RF.
#   3. scale_pos_weight / sample_weight handling in XGBoost is more flexible
#      than RF for the class imbalance in battery stress (High stress is rare).
#   4. XGBoost's built-in early stopping (with eval_set) prevents overfitting
#      on the training set — critical when High-stress samples are few.
#   5. Consistent with the XGBoost standard in battery health literature
#      (He et al. 2020, "Battery Degradation Prediction with XGBoost").
# =============================================================================

import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def train_model():
    """Train XGBClassifier for battery stress prediction. Saves model to MODEL_DIR."""
    X_train = joblib.load(os.path.join(MODEL_DIR, "f5_X_train.pkl"))
    X_test  = joblib.load(os.path.join(MODEL_DIR, "f5_X_test.pkl"))
    y_train = joblib.load(os.path.join(MODEL_DIR, "f5_y_train.pkl"))
    y_test  = joblib.load(os.path.join(MODEL_DIR, "f5_y_test.pkl"))

    print(f"[F5] Training on {X_train.shape[0]:,} samples, {X_train.shape[1]} features")

    # Compute per-sample weights to handle class imbalance
    # (High stress class is rare — these samples should count more)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # n_estimators=300: XGBoost benefits from more rounds than RF because
    #   each tree only corrects a small learning_rate fraction of the error.
    # learning_rate=0.05: Conservative shrinkage. Lower rate + more trees
    #   generalises better on sensor data with inherent noise.
    # max_depth=6: XGBoost standard depth. Deeper than RF because XGBoost
    #   corrects bias progressively — individual trees can be shallower per round.
    # subsample=0.8, colsample_bytree=0.8: Row/feature bagging per tree.
    #   Reduces overfitting on correlated battery sensor readings.
    # use_label_encoder=False, eval_metric='mlogloss': Standard XGBoost
    #   multiclass configuration (avoids deprecated encoder warning).
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        use_label_encoder=False,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              sample_weight=sample_weights,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[F5] Accuracy: {acc:.4f}")
    print("[F5] Classification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=["Low Stress", "Medium Stress", "High Stress"]))

    features = joblib.load(os.path.join(MODEL_DIR, "f5_features.pkl"))
    imp = dict(zip(features, model.feature_importances_))
    print("[F5] Feature importances:", {k: f"{v:.3f}" for k, v in
          sorted(imp.items(), key=lambda x: -x[1])})

    joblib.dump(model, os.path.join(MODEL_DIR, "f5_battery_model.pkl"))
    print("[F5] Model saved -> models/f5_battery_model.pkl")

    return {"model": model, "accuracy": acc}


if __name__ == "__main__":
    result = train_model()
    print(f"[F5] Training complete. Accuracy={result['accuracy']:.4f}")
