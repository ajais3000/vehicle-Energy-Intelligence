# =============================================================================
# FEATURE 1: Predictive HVAC Energy Load Optimizer
# FILE: features/feature1_hvac_optimizer/ml_model.py
#
# PURPOSE:
#   Trains a Random Forest Regressor to predict total HVAC power draw (AC +
#   Heater Watts) from ambient and vehicle conditions. Saves the trained model
#   as a .pkl file for use in the Streamlit dashboard.
#
# WHY RANDOM FOREST REGRESSOR?
#   1. HVAC load has strongly non-linear relationships with temperature —
#      the U-shaped demand curve (high at cold AND hot extremes) cannot be
#      captured by linear regression.
#   2. Random Forest is robust to the mix of unit scales in our features
#      (temperature vs. elevation vs. speed) even after standard scaling.
#   3. It provides feature_importances_ which confirms our domain intuition
#      that OAT is the dominant driver.
#   4. RF is resistant to overfitting on noisy HVAC sensor readings.
#
# ALTERNATIVE CONSIDERED: Gradient Boosting Regressor
#   GBR would give similar accuracy but is slower to train on 500 000+ rows
#   and provides less interpretability. RF with n_estimators=200 is the
#   better engineering trade-off here.
# =============================================================================

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def train_model():
    """
    Load preprocessed training data, fit the Random Forest, evaluate on test
    set, and persist the model. Returns the trained model and test metrics.
    """
    # ── Load preprocessed data ─────────────────────────────────────────────
    X_train = joblib.load(os.path.join(MODEL_DIR, "f1_X_train.pkl"))
    X_test  = joblib.load(os.path.join(MODEL_DIR, "f1_X_test.pkl"))
    y_train = joblib.load(os.path.join(MODEL_DIR, "f1_y_train.pkl"))
    y_test  = joblib.load(os.path.join(MODEL_DIR, "f1_y_test.pkl"))

    print(f"[F1] Training on {X_train.shape[0]:,} samples, {X_train.shape[1]} features")

    # ── Train ─────────────────────────────────────────────────────────────
    # n_estimators=200: More trees reduce variance. Diminishing returns after
    #   ~300 trees but negligible cost to go to 200 from 100.
    # max_depth=12: Enough depth to learn temperature-speed-gradient interactions
    #   without memorising sensor noise in individual second-by-second readings.
    # min_samples_leaf=5: Prevents leaves with tiny sample counts from noise-fitting.
    # n_jobs=-1: Use all CPU cores for parallel tree building.
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"[F1] HVAC Optimizer -> R^2: {r2:.4f}  |  RMSE: {rmse:.2f} Watts")

    # Feature importance log — confirms OAT should dominate
    features = joblib.load(os.path.join(MODEL_DIR, "f1_features.pkl"))
    imp = dict(zip(features, model.feature_importances_))
    print("[F1] Feature importances:", {k: f"{v:.3f}" for k, v in
          sorted(imp.items(), key=lambda x: -x[1])})

    # ── Save ────────────────────────────────────────────────────────────────
    joblib.dump(model, os.path.join(MODEL_DIR, "f1_hvac_model.pkl"))
    print("[F1] Model saved -> models/f1_hvac_model.pkl")

    return {"model": model, "r2": r2, "rmse": rmse}


# ── Standalone execution ───────────────────────────────────────────────────────
if __name__ == "__main__":
    result = train_model()
    print(f"[F1] Training complete. R^2={result['r2']:.4f}")
