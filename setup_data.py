"""
setup_data.py — Run ONCE locally to split the obesity dataset.

Usage:
    python setup_data.py --input ObesityDataSet_cleaned.csv
"""

import os
import argparse
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="ObesityDataSet_cleaned.csv")
    args = parser.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    test_size   = params["data"]["test_size"]
    random_seed = params["data"]["random_seed"]

    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"  Shape : {df.shape}")
    print(f"  Target distribution:\n{df['NObeyesdad'].value_counts()}")

    # Stratified split
    train_df, test_df = train_test_split(
        df, test_size=test_size,
        random_state=random_seed,
        stratify=df["NObeyesdad"]
    )

    print(f"\n  Train : {len(train_df)} rows")
    print(f"  Test  : {len(test_df)} rows")

    os.makedirs("train", exist_ok=True)
    os.makedirs("test",  exist_ok=True)
    os.makedirs("data",  exist_ok=True)

    train_df.to_csv("train/train.csv", index=False)
    test_df.to_csv("test/test.csv",   index=False)
    pd.DataFrame(columns=df.columns).to_csv("data/new_data.csv", index=False)

    print("\ntrain/train.csv saved")
    print("test/test.csv saved")
    print("data/new_data.csv placeholder created")
    print("\nNext steps:")
    print("dvc init")
    print("dvc add train/train.csv test/test.csv")
    print("git add train/train.csv.dvc test/test.csv.dvc .gitignore")
    print("git commit -m 'feat: track data with DVC'")
    print("dvc remote add -d origin https://dagshub.com/USER/REPO.dvc")
    print("dvc push")
    print("dvc repro")

if __name__ == "__main__":
    main()
