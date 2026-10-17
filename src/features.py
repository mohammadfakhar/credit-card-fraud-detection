import pandas as pd


def compute_behavioral_features(
    df: pd.DataFrame, card_mean_map: dict = None
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    # مرتب‌سازی زمانی
    df.sort_values(by=["cc_num", "trans_date_trans_time"], inplace=True)

    # ۱. فاصله زمانی از تراکنش قبلی (بر حسب ثانیه)
    df["prev_trans_time"] = df.groupby("cc_num")[
        "trans_date_trans_time"
    ].shift(1)
    df["time_diff"] = (
        (df["trans_date_trans_time"] - df["prev_trans_time"]).dt.total_seconds()
    ).fillna(999999)
    df.drop(columns=["prev_trans_time"], inplace=True)

    # ۲. نسبت مبلغ فعلی به میانگین تراکنش‌های کارت
    if card_mean_map is None:
        card_mean_map = df.groupby("cc_num")["amt"].mean().to_dict()

    global_mean = df["amt"].mean()
    df["card_mean_amt"] = (
        df["cc_num"].map(card_mean_map).fillna(global_mean)
    )
    df["amt_to_mean_ratio"] = df["amt"] / (df["card_mean_amt"] + 1e-5)
    df.drop(columns=["card_mean_amt"], inplace=True)

    return df, card_mean_map
