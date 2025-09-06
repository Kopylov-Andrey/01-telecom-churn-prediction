from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    print("Columns as-is:")
    for i, c in enumerate(df.columns, 1):
        print(f"{i:2d}. {repr(c)}")


if __name__ == "__main__":
    main()
