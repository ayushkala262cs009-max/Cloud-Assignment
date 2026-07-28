import pandas as pd
meta = pd.read_csv(r"D:\highD-dataset-v1.0\data\01_tracksMeta.csv")
print(meta.columns.tolist())
print(meta.head(10))