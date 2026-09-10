import pandas as pd
df = pd.read_csv("data/processed/triage_results.csv")
print(df.sort_values("pct_usable_tiles"))