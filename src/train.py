import json
import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from data_preprocessing import preprocess_data
from features import compute_behavioral_features


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    print("[+] Loading raw data...")
    df = pd.read_csv(cfg["data"]["raw_train_path"])

    print("[+] Preprocessing base features...")
    df = preprocess_data(df)

    # جداسازی متغیر هدف و حذف فیلدهای بدون نیاز در ماتریس ویژگی
    target_col = cfg["model"]["target_col"]
    y = df[target_col]
    X = df.drop(columns=[target_col, "trans_date_trans_time"])

    print("[+] Train / Validation split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=cfg["params"]["random_state"], stratify=y
    )

    print("[+] Engineering behavioral features...")
    # بازگرداندن ستون تاریخ موقتاً برای محاسبات فیچرهای زمانی
    X_train_full = X_train.copy()
    X_train_full["trans_date_trans_time"] = df.loc[X_train.index, "trans_date_trans_time"]
    X_train_feat, card_means = compute_behavioral_features(X_train_full)
    X_train_feat.drop(columns=["trans_date_trans_time", "cc_num"], inplace=True)

    X_val_full = X_val.copy()
    X_val_full["trans_date_trans_time"] = df.loc[X_val.index, "trans_date_trans_time"]
    X_val_feat, _ = compute_behavioral_features(X_val_full, card_mean_map=card_means)
    X_val_feat.drop(columns=["trans_date_trans_time", "cc_num"], inplace=True)

    # همگام‌سازی شاخص‌های y با ترتیب X پس از مرتب‌سازی زمانی در compute_behavioral_features
    y_train_feat = y.loc[X_train_feat.index]
    y_val_feat = y.loc[X_val_feat.index]

    print("[+] Training LightGBM model...")
    p = cfg["params"]
    model = LGBMClassifier(
        objective=p["objective"],
        metric=p["metric"],
        boosting_type=p["boosting_type"],
        learning_rate=p["learning_rate"],
        num_leaves=p["num_leaves"],
        scale_pos_weight=p["scale_pos_weight"],
        random_state=p["random_state"],
        n_estimators=p["n_estimators"],
        n_jobs=-1,
    )

    model.fit(
        X_train_feat,
        y_train_feat,
        eval_set=[(X_val_feat, y_val_feat)],
        callbacks=[],
    )

    print("[+] Evaluating on validation set...")
    val_preds_prob = model.predict_proba(X_val_feat)[:, 1]
    pr_auc = average_precision_score(y_val_feat, val_preds_prob)
    roc_auc = roc_auc_score(y_val_feat, val_preds_prob)
    print(f"Validation PR-AUC:  {pr_auc:.4f}")
    print(f"Validation ROC-AUC: {roc_auc:.4f}")

    # بهینه‌سازی آستانه تصمیم‌گیری (Threshold Tuning)
    precisions, recalls, thresholds = precision_recall_curve(y_val_feat, val_preds_prob)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = float(thresholds[best_idx])
    print(f"Optimal Threshold: {best_thresh:.4f} (Max F1: {f1_scores[best_idx]:.4f})")

    # ذخیره مدل و آرتیفکت‌ها
    print("[+] Saving model and metadata...")
    artifacts = {
        "model": model,
        "card_means": card_means,
        "features": list(X_train_feat.columns),
    }
    joblib.dump(artifacts, cfg["model"]["output_path"])

    with open(cfg["model"]["threshold_path"], "w") as f:
        json.dump({"threshold": best_thresh}, f, indent=4)

    print("[✓] Training completed successfully!")


if __name__ == "__main__":
    main()
