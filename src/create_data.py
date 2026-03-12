import pandas as pd

df = pd.read_csv("data/housing.csv")

df_small = df.head(5000)

df_small.to_csv("data/housing.csv", index=False)

print("Dataset Version 1 created with 5000 rows")