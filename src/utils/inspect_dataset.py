import pandas as pd
import numpy as np

data_path = "data/exoTrain.csv"
df = pd.read_csv(data_path)
print("Dataset loaded successfully.\n")
print("Shape of dataset:", df.shape)
print("\nFirst 5 columns:", df.columns[:5])
print("Last 5 columns:", df.columns[-5:])

label_counts = df["LABEL"].value_counts()
print("\nLabel distribution:")
print(label_counts)
print("\nLabel percentages:")
print(label_counts / len(df) * 100)

print("\nTotal missing values:", df.isnull().sum().sum())
flux_values = df.drop("LABEL", axis=1).values
print("\nFlux min:", np.min(flux_values))
print("Flux max:", np.max(flux_values))
print("Flux mean:", np.mean(flux_values))
print("\nExample row where LABEL=2:")
print(df[df["LABEL"] == 2].iloc[0, :10])