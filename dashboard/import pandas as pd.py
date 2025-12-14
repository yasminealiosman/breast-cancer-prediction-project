import pandas as pd

csv_path = r"C:\Users\yasmine\Documents\Portfolio\DataSciencePortfolio\Projects\Breast-Cancer\dashboard\X_test.csv"

try:
    df = pd.read_csv(csv_path)
    print("✅ File loaded successfully")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head())
except Exception as e:
    print("❌ Failed to load CSV:", e)