from pathlib import Path
import requests

URL = (
    "https://www.kaggle.com/datasets/blastchar/telco-customer-churn/croissant/download"
)


def main():
    root = Path(__file__).resolve().parents[1]  # корень проекта
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "telco.csv"

    print(f"Downloading dataset to {dest} ...")
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"Saved: {dest} ({len(resp.content)} bytes)")


if __name__ == "__main__":
    print("2")
    main()
