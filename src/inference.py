import json
import joblib
import pandas as pd
from data_preprocessing import preprocess_data


class FraudDetector:
    def __init__(
        self,
        model_path="models/lgbm_model.pkl",
        threshold_path="models/threshold.json",
    ):
        artifacts = joblib.load(model_path)
        self.model = artifacts["model"]
        self.card_means = artifacts["card_means"]
        self.feature_names = artifacts["features"]

        with open(threshold_path, "r") as f:
            self.threshold = json.load(f)["threshold"]

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = preprocess_data(raw_df)

        # استخراج ویژگی نسبت به میانگین کارت
        global_mean = sum(self.card_means.values()) / max(len(self.card_means), 1)
        card_mean_amt = df["cc_num"].map(self.card_means).fillna(global_mean)
        df["amt_to_mean_ratio"] = df["amt"] / (card_mean_amt + 1e-5)

        # در حالت تک تراکنش استنتاجی، در صورت نبود تاریخچه فاصله را پیش‌فرض می‌گذاریم
        if "time_diff" not in df.columns:
            df["time_diff"] = 999999.0

        # تطبیق دقیق ستون‌ها با مدل
        X = df[self.feature_names]

        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        results = raw_df.copy()
        results["fraud_probability"] = probs
        results["is_fraud_predicted"] = preds
        return results[["trans_num", "fraud_probability", "is_fraud_predicted"]]


if __name__ == "__main__":
    detector = FraudDetector()
    print(f"[✓] FraudDetector loaded with threshold: {detector.threshold:.4f}")
