import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the dataset
# Note: You need to download the dataset from Kaggle first

df = pd.read_csv('dataset_final.csv')

print("=" * 60)
print("STEP 1: DATA EXPLORATION")
print("=" * 60)

# Basic information
print("\n1. Dataset Shape:")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n2. Column Names:")
print(f"   {df.columns.tolist()}")

print("\n3. First Few Rows:")
print(df.head())

print("\n4. Data Types:")
print(df.dtypes)

print("\n5. Missing Values:")
print(df.isnull().sum())

print("\n6. Statistical Summary:")
print(df.describe())

# Check for the four sensor modalities mentioned in the paper
print("\n7. Sensor Modalities (Expected: temperature, humidity, light, loudness):")
sensor_columns = [col for col in df.columns if any(
    sensor in col.lower() for sensor in ['temp', 'humid', 'light', 'loud', 'sound']
)]
print(f"   Found: {sensor_columns}")

# Time information if available
if 'timestamp' in df.columns or 'date' in df.columns or 'time' in df.columns:
    time_col = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()][0]
    print(f"\n8. Time Range:")
    print(f"   Start: {df[time_col].min()}")
    print(f"   End: {df[time_col].max()}")