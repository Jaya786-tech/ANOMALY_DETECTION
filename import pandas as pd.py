import pandas as pd
from src.models.ensemble import HybridEnsemble

def evaluate_file(csv_path):
    df = pd.read_csv(csv_path)
    hybrid = HybridEnsemble("models")

    if "label" in df.columns:
        return hybrid.evaluate(df, df["label"])
    return hybrid.evaluate(df)

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python3 -m src.evaluate dataset.csv")
    else:
        out = evaluate_file(sys.argv[1])
        print(json.dumps(out, indent=2))
