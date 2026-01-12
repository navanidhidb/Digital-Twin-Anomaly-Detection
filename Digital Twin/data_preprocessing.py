import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import uniform_filter1d

print("=" * 60)
print("STEP 2: DATA PREPROCESSING")
print("=" * 60)

# Load the dataset
df = pd.read_csv('dataset_final.csv')

# Print all column names to see what's available
print("\n1. Available columns in dataset:")
print(df.columns.tolist())

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Find sensor columns automatically
sensor_cols = []
for col in df.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in ['temp', 'humid', 'light', 'loud', 'sound', 'noise']):
        sensor_cols.append(col)

# If no sensor columns found, use all numeric columns
if len(sensor_cols) == 0:
    print("\nWarning: No sensor columns found. Using all numeric columns.")
    sensor_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n2. Selected sensor columns: {sensor_cols}")

# Extract sensor data
sensor_data = df[sensor_cols].copy()

print("\n3. Original Data Statistics:")
print(sensor_data.describe())

# Check for missing values
print("\n4. Missing Values Check:")
missing_values = sensor_data.isnull().sum()
print(missing_values)

if missing_values.sum() > 0:
    print("\n   Handling missing values using forward fill...")
    sensor_data = sensor_data.fillna(method='ffill').fillna(method='bfill')
    print("   [OK] Missing values handled")

# Min-Max Scaling (0-1 normalization)
print("\n5. Applying Min-Max Scaling (0-1)...")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(sensor_data)
scaled_df = pd.DataFrame(scaled_data, columns=sensor_cols)

print("   Scaled Data Statistics:")
print(scaled_df.describe())

# Moving Average Smoothing
print("\n6. Applying Moving Average Smoothing...")
WINDOW_SIZE = 5  # 5-point moving average

smoothed_data = np.zeros_like(scaled_data)
for i in range(scaled_data.shape[1]):
    smoothed_data[:, i] = uniform_filter1d(scaled_data[:, i], size=WINDOW_SIZE, mode='nearest')

smoothed_df = pd.DataFrame(smoothed_data, columns=sensor_cols)

print(f"   Window size: {WINDOW_SIZE}")
print("   Smoothed Data Statistics:")
print(smoothed_df.describe())

# Save preprocessed data
output_file = 'data_preprocessed.csv'
smoothed_df.to_csv(output_file, index=False)
print(f"\n7. [SUCCESS] Preprocessed data saved to '{output_file}'")
print(f"   Shape: {smoothed_df.shape}")
print(f"   Columns: {list(smoothed_df.columns)}")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)
