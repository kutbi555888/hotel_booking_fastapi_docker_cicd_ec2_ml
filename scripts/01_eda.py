from __future__ import annotations


from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.hotel_booking_ml.config import EDA_REPORT_DIR, MONTH_ORDER, TARGET_COLUMN
from src.hotel_booking_ml.data.dataset import load_raw_data, validate_dataset
from src.hotel_booking_ml.evaluation.plots import (
    save_boxplot_by_target,
    save_missing_values_plot,
    save_monthly_cancellation_rate_plot,
    save_target_distribution_plot,
)
from src.hotel_booking_ml.utils.io import ensure_project_directories, save_json
from src.hotel_booking_ml.utils.logger import setup_logger

logger = setup_logger()

def main() -> None:
    ensure_project_directories()
    df = load_raw_data()
    validate_dataset(df)

    missing_df = (
        pd.DataFrame(
            {
                "column": df.columns,
                "missing_count": df.isna().sum().values,
                "missing_pct": (df.isna().mean() * 100).round(2).values,
            }
        )
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )

    numeric_summary = df.describe(include=["number"]).T
    categorical_summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "nunique": df.nunique(dropna=False),
            "top_value": [df[col].mode(dropna=False).iloc[0] if not df[col].mode(dropna=False).empty else None for col in df.columns],
        }
    )

    target_distribution = (
        df[TARGET_COLUMN].value_counts(normalize=True).sort_index().rename("ratio").reset_index()
    )
    target_distribution["ratio"] = (target_distribution["ratio"] * 100).round(4)

    month_cancel_rate = (
        df.groupby("arrival_date_month")[TARGET_COLUMN]
        .mean()
        .reindex(MONTH_ORDER)
        .reset_index()
        .rename(columns={TARGET_COLUMN: "cancellation_rate"})
    )

    overview = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "zero_guest_rows": int(
            ((df["adults"] + df["children"].fillna(0) + df["babies"]) == 0).sum()
        ),
        "negative_adr_rows": int((df["adr"] < 0).sum()),
        "constant_columns": [col for col in df.columns if df[col].nunique(dropna=False) == 1],
        "target_distribution_pct": {
            str(int(key)): float(value)
            for key, value in (df[TARGET_COLUMN].value_counts(normalize=True) * 100).sort_index().items()
        },
        "top_missing_columns_pct": {
            row["column"]: float(row["missing_pct"])
            for _, row in missing_df.head(5).iterrows()
        },
        "notes": [
            "company ustunida missing juda ko‘p, shuning uchun modelingda bu ustundan faqat has_company kabi signal olinadi.",
            "arrival_date_year bitta qiymatdan iborat, shuning uchun u predictive signal bermaydi.",
            "reservation_status va reservation_status_date natijadan keyingi ma’lumot bo‘lishi mumkin, shu sababli leakage sifatida tashlanadi.",
        ],
    }

    EDA_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    missing_df.to_csv(EDA_REPORT_DIR / "missing_values.csv", index=False)
    numeric_summary.to_csv(EDA_REPORT_DIR / "numeric_summary.csv")
    categorical_summary.to_csv(EDA_REPORT_DIR / "categorical_summary.csv")
    target_distribution.to_csv(EDA_REPORT_DIR / "target_distribution.csv", index=False)
    month_cancel_rate.to_csv(EDA_REPORT_DIR / "monthly_cancellation_rate.csv", index=False)
    save_json(overview, EDA_REPORT_DIR / "overview.json")

    save_target_distribution_plot(df[TARGET_COLUMN], EDA_REPORT_DIR / "target_distribution.png")
    save_missing_values_plot(missing_df, EDA_REPORT_DIR / "missing_top10.png")
    save_boxplot_by_target(df, "lead_time", TARGET_COLUMN, EDA_REPORT_DIR / "lead_time_by_target.png")
    save_boxplot_by_target(df, "adr", TARGET_COLUMN, EDA_REPORT_DIR / "adr_by_target.png")
    save_monthly_cancellation_rate_plot(month_cancel_rate, EDA_REPORT_DIR / "monthly_cancellation_rate.png")

    summary_md = f"""# EDA xulosasi

## Dataset haqida
- Qatorlar soni: **{df.shape[0]:,}**
- Ustunlar soni: **{df.shape[1]}**
- Dublikat qatorlar: **{int(df.duplicated().sum())}**
- Target ustuni: **{TARGET_COLUMN}**

## Muhim kuzatuvlar
- Cancel bo‘lmagan booking ulushi: **{overview["target_distribution_pct"]["0"]:.2f}%**
- Cancel bo‘lgan booking ulushi: **{overview["target_distribution_pct"]["1"]:.2f}%**
- `company` missing: **{missing_df.loc[missing_df["column"] == "company", "missing_pct"].iloc[0]:.2f}%**
- `agent` missing: **{missing_df.loc[missing_df["column"] == "agent", "missing_pct"].iloc[0]:.2f}%**
- `country` missing: **{missing_df.loc[missing_df["column"] == "country", "missing_pct"].iloc[0]:.2f}%**
- `children` ustunida juda kam missing bor: **{int(df["children"].isna().sum())} ta**
- `arrival_date_year` faqat bitta qiymatga ega: **{df["arrival_date_year"].iloc[0]}**
- Nol mehmonli bookinglar: **{overview["zero_guest_rows"]} ta**
- Manfiy ADR qiymatlari: **{overview["negative_adr_rows"]} ta**

## Modelingdan oldingi qarorlar
1. `reservation_status` va `reservation_status_date` leakage bo‘lishi mumkin, shu sababli modelga berilmaydi.
2. `arrival_date_year` constant bo‘lgani uchun tashlanadi.
3. `company` ustunining o‘zi tashlanadi, lekin undan `has_company` feature olinadi.
4. `hotel` ustunidan `hotel_type` feature olinadi, keyin original `hotel` tashlanadi.
5. Qo‘shimcha ravishda `total_nights`, `total_guests`, `is_family`, `room_changed` kabi engineered feature lar yasaladi.

## Saqlangan fayllar
- `missing_values.csv`
- `numeric_summary.csv`
- `categorical_summary.csv`
- `target_distribution.csv`
- `monthly_cancellation_rate.csv`
- Grafiklar (`.png`)
"""
    (EDA_REPORT_DIR / "eda_summary.md").write_text(summary_md, encoding="utf-8")

    logger.info("EDA tugadi. Natijalar %s papkaga saqlandi.", EDA_REPORT_DIR)

if __name__ == "__main__":
    main()
