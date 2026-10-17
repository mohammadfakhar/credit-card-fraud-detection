import numpy as np
import pandas as pd


def haversine_np(lat1, lon1, lat2, lon2):
    """محاسبه فاصله جغرافیایی با فرمول هاورسین به صورت وکتورایز شده"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # حذف ستون‌های شناسه و کم‌ارزش
    drop_cols = [
        col
        for col in [
            "Unnamed: 0",
            "trans_num",
            "first",
            "last",
            "street",
            "unix_time",
        ]
        if col in df.columns
    ]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # استخراج فیچرهای زمانی
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["age"] = (
        (df["trans_date_trans_time"] - df["dob"]).dt.days // 365.25
    ).astype(int)

    # فاصله مبدا تا مقصد تراکنش
    df["distance_km"] = haversine_np(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # تبدیل ستون‌های متنی به نوع categorical
    cat_cols = ["category", "gender", "job", "state", "merchant"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df

