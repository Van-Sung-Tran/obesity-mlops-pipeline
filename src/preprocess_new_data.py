"""
preprocess_new_data.py — COS40007 Group Task 3 (Studio 9)
Obesity Level Prediction — MLOps Retraining Pipeline

Checks if new_data.csv has new rows and appends them to train/train.csv.
Run BEFORE model.py in the retraining pipeline.

Steps:
  1. Load data/new_data.csv
  2. Validate schema matches train/train.csv
  3. Append new rows to train/train.csv
  4. Log how many rows were added
  5. Save data version metadata
"""

import os
import sys
import json
import logging
import datetime
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocessing.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_DATA  = os.path.join(ROOT, "data",  "new_data.csv")
TRAIN_CSV = os.path.join(ROOT, "train", "train.csv")
META_FILE = os.path.join(ROOT, "data_version.txt")


def main():
    log.info("=" * 55)
    log.info("  PREPROCESS NEW DATA — COS40007 Group Task 3")
    log.info("=" * 55)

    # ── Check new_data.csv exists ─────────────────────────────────────────────
    if not os.path.exists(NEW_DATA):
        log.info("  No new_data.csv found — skipping preprocessing")
        log.info("  Training will use existing train/train.csv")
        return

    # ── Load files ────────────────────────────────────────────────────────────
    log.info(f"\n[1/4] Loading data files...")
    new_df   = pd.read_csv(NEW_DATA)
    train_df = pd.read_csv(TRAIN_CSV)

    log.info(f"  new_data.csv : {len(new_df)} rows x {len(new_df.columns)} cols")
    log.info(f"  train.csv    : {len(train_df)} rows x {len(train_df.columns)} cols")

    # ── Check if new_data.csv is empty ────────────────────────────────────────
    if len(new_df) == 0:
        log.info("  new_data.csv is empty — skipping preprocessing")
        return

    # ── Validate schema ───────────────────────────────────────────────────────
    log.info(f"\n[2/4] Validating schema...")
    missing_cols = [c for c in train_df.columns if c not in new_df.columns]
    extra_cols   = [c for c in new_df.columns   if c not in train_df.columns]

    if missing_cols:
        log.error(f"  Missing columns in new_data.csv: {missing_cols}")
        log.error("  Skipping preprocessing — schema mismatch")
        return

    if extra_cols:
        log.warning(f"  Extra columns in new_data.csv (will be dropped): {extra_cols}")
        new_df = new_df[train_df.columns]

    log.info(f"  Schema validation passed ✓")

    # ── Append new data to train.csv ──────────────────────────────────────────
    log.info(f"\n[3/4] Appending {len(new_df)} new rows to train/train.csv...")
    combined_df = pd.concat([train_df, new_df], ignore_index=True)

    # Remove duplicates
    before_dedup = len(combined_df)
    combined_df  = combined_df.drop_duplicates()
    dupes_removed = before_dedup - len(combined_df)

    if dupes_removed > 0:
        log.info(f"  Removed {dupes_removed} duplicate rows")

    combined_df.to_csv(TRAIN_CSV, index=False)
    log.info(f"  train.csv updated: {len(train_df)} → {len(combined_df)} rows")
    log.info(f"  New rows added: {len(combined_df) - len(train_df)}")

    # ── Clear new_data.csv (keep headers only) ────────────────────────────────
    log.info(f"\n[4/4] Clearing new_data.csv (keeping headers)...")
    pd.DataFrame(columns=train_df.columns).to_csv(NEW_DATA, index=False)
    log.info(f"  new_data.csv reset — ready for next batch")

    # ── Save data version metadata ────────────────────────────────────────────
    version = 1
    if os.path.exists(META_FILE):
        try:
            version = int(open(META_FILE).read().strip()) + 1
        except ValueError:
            version = 1

    with open(META_FILE, "w") as f:
        f.write(str(version))

    # Save update log
    update_log = {
        "timestamp":    datetime.datetime.utcnow().isoformat(),
        "data_version": version,
        "rows_added":   len(new_df),
        "total_rows":   len(combined_df),
        "dupes_removed": dupes_removed,
    }
    with open("data_update_log.json", "w") as f:
        json.dump(update_log, f, indent=2)

    log.info(f"\n  ✓ Preprocessing complete.")
    log.info(f"  Data version: v{version}")
    log.info(f"  Total training rows: {len(combined_df)}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
