# =============================================================================
# FEATURE 2: Gradient-Aware Regenerative Braking Intensity Predictor
# FILE: features/feature2_regen_braking/ml_model.py
#
# PURPOSE:
#   Trains a Gradient Boosting Regressor to predict HV Battery Current[A]
#   during deceleration. Negative predictions = regen braking is occurring.
#
# WHY GRADIENT BOOSTING REGRESSOR?
#   1. GBR excels at capturing the multi-way interaction between gradient,
#      speed, and SOC that determines regen magnitude. These relationships
#      are sequential (gradient -> speed change -> current), which GBR's
#      stagewise additive approach handles well.
#   2. GBR typically outperforms RF for regression on structured tabular
#      data with interaction effects (established in Friedman 2001).
#   3. GBR provides reliable uncertainty via out-of-bag estimates and
#      staged_predict — useful for future confidence interval extensions.
#
# ALTERNATIVE CONSIDERED: XGBoost
#   XGBoost would be marginally faster but adds a dependency. sklearn's
#   GBR with the same hyperparameters achieves comparable R^2 here.
# =============================================================================

import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def train_model():
    """Train GBR for regen braking prediction. Saves model to MODEL_DIR."""
    X_train = joblib.load(os.path.join(MODEL_DIR, "f2_X_train.pkl"))
    X_test  = joblib.load(os.path.join(MODEL_DIR, "f2_X_test.pkl"))
    y_train = joblib.load(os.path.join(MODEL_DIR, "f2_y_train.pkl"))
    y_test  = joblib.load(os.path.join(MODEL_DIR, "f2_y_test.pkl"))

    print(f"[F2] Training on {X_train.shape[0]:,} samples")

    # n_estimators=200: Sequential boosted trees. 200 is sufficient for this
    #   feature count; more increases training time linearly.
    # learning_rate=0.1: Standard shrinkage. Prevents single powerful trees
    #   from dominating — key for stability on sensor noise.
    # max_depth=5: GBR trees should be shallow (3–6). Deeper trees overfit
    #   because of the sequential additive nature of boosting.
    # subsample=0.8: Stochastic GBR — randomly sample 80% of training rows
    #   per tree. Reduces variance and speeds up training on large datasets.
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"[F2] Regen Braking -> R^2: {r2:.4f}  |  RMSE: {rmse:.2f} A")

    features = joblib.load(os.path.join(MODEL_DIR, "f2_features.pkl"))
    imp = dict(zip(features, model.feature_importances_))
    print("[F2] Feature importances:", {k: f"{v:.3f}" for k, v in
          sorted(imp.items(), key=lambda x: -x[1])})

    joblib.dump(model, os.path.join(MODEL_DIR, "f2_regen_model.pkl"))
    print("[F2] Model saved -> models/f2_regen_model.pkl")

    return {"model": model, "r2": r2, "rmse": rmse}


if __name__ == "__main__":
    result = train_model()
    print(f"[F2] Training complete. R^2={result['r2']:.4f}")
