import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

print("=" * 60)
print("STEP 6: SENSOR DATA VISUALIZATION WITH ANOMALIES")
print("=" * 60)

# Load original data and hybrid results
df_original = pd.read_csv('dataset_final.csv')
hybrid_results = pd.read_csv('hybrid_results.csv')

# Identify sensor columns (adjust based on actual dataset)
# Common patterns: temperature, temp, humidity, humid, light, loudness, sound, noise
sensor_cols = []
for col in df_original.columns:
    if any(keyword in col.lower() for keyword in ['temp', 'humid', 'light', 'loud', 'sound', 'noise']):
        sensor_cols.append(col)

# If we can't find sensor columns, use the preprocessed data column names
if len(sensor_cols) == 0:
    print("Warning: Could not identify sensor columns. Using default names.")
    sensor_cols = ['temperature', 'humidity', 'light', 'loudness']
    # Create dummy dataframe with sensor data
    preprocessed_data = pd.read_csv('data_preprocessed.csv')
    df_sensors = preprocessed_data.copy()
    df_sensors.columns = sensor_cols
else:
    df_sensors = df_original[sensor_cols].copy()

print(f"\n1. Identified Sensor Columns: {sensor_cols}")

# Get hybrid labels and scores
hybrid_labels = hybrid_results['hybrid_label'].values
hybrid_scores = hybrid_results['hybrid_score'].values

# Ensure length matches
min_len = min(len(df_sensors), len(hybrid_labels))
df_sensors = df_sensors.iloc[:min_len]
hybrid_labels = hybrid_labels[:min_len]
hybrid_scores = hybrid_scores[:min_len]

# Check if timestamp column exists
timestamp_col = None
for col in df_original.columns:
    if 'time' in col.lower() or 'date' in col.lower():
        timestamp_col = col
        break

if timestamp_col:
    timestamps = pd.to_datetime(df_original[timestamp_col].iloc[:min_len])
    time_axis = timestamps
    time_label = 'Timestamp'
else:
    time_axis = np.arange(min_len)
    time_label = 'Time Index'

print(f"2. Time axis type: {time_label}")
print(f"3. Data points to visualize: {min_len}")

# Figure 2 from paper: IoT Data with Hybrid Anomaly Detection
fig, axes = plt.subplots(4, 1, figsize=(16, 12))
fig.suptitle('IoT Sensor Data with Hybrid Anomaly Detection\n(Similar to Figure 2 in Paper)',
             fontsize=16, fontweight='bold')

# Color map for anomaly intensity (white to red)
cmap = LinearSegmentedColormap.from_list('anomaly', ['white', 'red'])

for idx, col in enumerate(sensor_cols[:4]):  # Plot first 4 sensors
    ax = axes[idx]
    
    # Plot normal data in blue
    normal_mask = hybrid_labels == 0
    ax.plot(time_axis[normal_mask], df_sensors[col].values[normal_mask],
            'o', color='steelblue', alpha=0.5, markersize=2, label='Normal')
    
    # Plot anomalies in red with intensity based on confidence score
    anomaly_mask = hybrid_labels == 1
    if anomaly_mask.sum() > 0:
        scatter = ax.scatter(time_axis[anomaly_mask],
                           df_sensors[col].values[anomaly_mask],
                           c=hybrid_scores[anomaly_mask],
                           cmap=cmap,
                           vmin=0.6, vmax=1.0,
                           s=30,
                           marker='o',
                           edgecolors='darkred',
                           linewidths=0.5,
                           label='Anomaly',
                           zorder=5)
        
        # Add colorbar for first subplot only
        if idx == 0:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
            cbar.set_label('Anomaly Confidence', rotation=270, labelpad=15)
    
    ax.set_ylabel(col.capitalize(), fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9)
    
    # Highlight major anomaly regions
    anomaly_regions = []
    in_region = False
    region_start = None
    
    for i in range(len(hybrid_labels)):
        if hybrid_labels[i] == 1 and hybrid_scores[i] >= 0.7:
            if not in_region:
                region_start = i
                in_region = True
        else:
            if in_region:
                anomaly_regions.append((region_start, i-1))
                in_region = False
    
    # Add background shading for anomaly regions
    for start, end in anomaly_regions:
        if end - start > 5:  # Only shade regions with multiple points
            ax.axvspan(time_axis[start], time_axis[end],
                      alpha=0.2, color='red', zorder=0)

axes[-1].set_xlabel(time_label, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('sensor_data_with_anomalies.png', dpi=300, bbox_inches='tight')
print("\n4. Sensor visualization saved as 'sensor_data_with_anomalies.png'")

# Identify key anomaly events (clusters of anomalies)
print("\n5. Identifying Major Anomaly Events:")

# Find consecutive anomaly sequences
anomaly_indices = np.where(hybrid_labels == 1)[0]

if len(anomaly_indices) > 0:
    # Group consecutive indices
    anomaly_events = []
    current_event = [anomaly_indices[0]]
    
    for i in range(1, len(anomaly_indices)):
        if anomaly_indices[i] - anomaly_indices[i-1] <= 10:  # Within 10 points
            current_event.append(anomaly_indices[i])
        else:
            if len(current_event) >= 3:  # At least 3 consecutive anomalies
                anomaly_events.append(current_event)
            current_event = [anomaly_indices[i]]
    
    if len(current_event) >= 3:
        anomaly_events.append(current_event)
    
    print(f"   Found {len(anomaly_events)} major anomaly events:")
    
    for event_num, event in enumerate(anomaly_events[:5], 1):  # Show top 5
        start_idx = event[0]
        end_idx = event[-1]
        avg_score = hybrid_scores[event].mean()
        
        if timestamp_col:
            start_time = timestamps.iloc[start_idx]
            end_time = timestamps.iloc[end_idx]
            print(f"\n   Event {event_num}:")
            print(f"     Time: {start_time} to {end_time}")
        else:
            print(f"\n   Event {event_num}:")
            print(f"     Index range: {start_idx} to {end_idx}")
        
        print(f"     Duration: {len(event)} data points")
        print(f"     Average confidence: {avg_score:.4f}")
        
        # Show sensor values during this event
        print(f"     Sensor values:")
        for col in sensor_cols[:4]:
            values = df_sensors[col].iloc[event].values
            print(f"       {col}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")

# Statistics by sensor
print(f"\n6. Anomaly Statistics by Sensor:")
for col in sensor_cols[:4]:
    anomaly_values = df_sensors[col].values[hybrid_labels == 1]
    normal_values = df_sensors[col].values[hybrid_labels == 0]
    
    print(f"\n   {col.capitalize()}:")
    print(f"     Normal: mean={normal_values.mean():.2f}, std={normal_values.std():.2f}")
    if len(anomaly_values) > 0:
        print(f"     Anomaly: mean={anomaly_values.mean():.2f}, std={anomaly_values.std():.2f}")
        print(f"     Difference: {abs(anomaly_values.mean() - normal_values.mean()):.2f}")

print("\n[SUCCESS] Sensor visualization completed successfully!")
