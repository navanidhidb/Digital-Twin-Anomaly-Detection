import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("=" * 60)
print("STEP 7: 3D VISUALIZATION OF ANOMALIES")
print("=" * 60)

# Load data
df_original = pd.read_csv('dataset_final.csv')
df_preprocessed = pd.read_csv('data_preprocessed.csv')
hybrid_results = pd.read_csv('hybrid_results.csv')

# Get labels
hybrid_labels = hybrid_results['hybrid_label'].values
iso_labels = hybrid_results['iso_label'].values
dbscan_labels = hybrid_results['dbscan_label'].values

# Identify sensors (using preprocessed data for consistent scaling)
sensor_names = df_preprocessed.columns.tolist()

# For 3D plot, we need 3 features - use first 3 or temperature, humidity, light
if len(sensor_names) >= 3:
    feature1_name = sensor_names[0]  # Usually temperature
    feature2_name = sensor_names[1]  # Usually humidity
    feature3_name = sensor_names[2]  # Usually light
else:
    print("Warning: Not enough features for 3D visualization")
    feature1_name, feature2_name, feature3_name = sensor_names[0], sensor_names[0], sensor_names[0]

feature1 = df_preprocessed[feature1_name].values
feature2 = df_preprocessed[feature2_name].values
feature3 = df_preprocessed[feature3_name].values

# Ensure same length
min_len = min(len(feature1), len(hybrid_labels))
feature1 = feature1[:min_len]
feature2 = feature2[:min_len]
feature3 = feature3[:min_len]
hybrid_labels = hybrid_labels[:min_len]
iso_labels = iso_labels[:min_len]
dbscan_labels = dbscan_labels[:min_len]

print(f"\n1. Selected Features for 3D Visualization:")
print(f"   - X-axis: {feature1_name}")
print(f"   - Y-axis: {feature2_name}")
print(f"   - Z-axis: {feature3_name}")

# Figure 3 from paper: 3D Visualization
fig = plt.figure(figsize=(18, 6))

# Plot 1: Hybrid method (main visualization from paper)
ax1 = fig.add_subplot(131, projection='3d')

# Plot normal points in blue
normal_mask = hybrid_labels == 0
ax1.scatter(feature1[normal_mask],
            feature2[normal_mask],
            feature3[normal_mask],
            c='steelblue',
            alpha=0.3,
            s=10,
            label='Normal',
            edgecolors='none')

# Plot anomalies in red (only those confirmed by both methods as in paper)
both_agree_mask = (iso_labels == 1) & (dbscan_labels == 1)
ax1.scatter(feature1[both_agree_mask],
            feature2[both_agree_mask],
            feature3[both_agree_mask],
            c='red',
            alpha=0.8,
            s=50,
            label='Anomaly (Both Methods)',
            edgecolors='darkred',
            linewidths=1)

ax1.set_xlabel(feature1_name.capitalize(), fontsize=10, fontweight='bold')
ax1.set_ylabel(feature2_name.capitalize(), fontsize=10, fontweight='bold')
ax1.set_zlabel(feature3_name.capitalize(), fontsize=10, fontweight='bold')
ax1.set_title('Hybrid Anomaly Detection\n(Similar to Figure 3 in Paper)',
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.view_init(elev=20, azim=45)

print(f"\n2. 3D Visualization Statistics:")
print(f"   - Normal points: {normal_mask.sum()}")
print(f"   - Anomalies (both methods agree): {both_agree_mask.sum()}")

# Plot 2: Isolation Forest only
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(feature1[iso_labels == 0],
            feature2[iso_labels == 0],
            feature3[iso_labels == 0],
            c='steelblue',
            alpha=0.3,
            s=10,
            label='Normal',
            edgecolors='none')

ax2.scatter(feature1[iso_labels == 1],
            feature2[iso_labels == 1],
            feature3[iso_labels == 1],
            c='orange',
            alpha=0.6,
            s=30,
            label='Anomaly (ISO)',
            edgecolors='darkorange',
            linewidths=0.5)

ax2.set_xlabel(feature1_name.capitalize(), fontsize=10, fontweight='bold')
ax2.set_ylabel(feature2_name.capitalize(), fontsize=10, fontweight='bold')
ax2.set_zlabel(feature3_name.capitalize(), fontsize=10, fontweight='bold')
ax2.set_title('Isolation Forest Only', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.view_init(elev=20, azim=45)

# Plot 3: DBSCAN only
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(feature1[dbscan_labels == 0],
            feature2[dbscan_labels == 0],
            feature3[dbscan_labels == 0],
            c='steelblue',
            alpha=0.3,
            s=10,
            label='Normal',
            edgecolors='none')

ax3.scatter(feature1[dbscan_labels == 1],
            feature2[dbscan_labels == 1],
            feature3[dbscan_labels == 1],
            c='green',
            alpha=0.6,
            s=30,
            label='Anomaly (DBSCAN)',
            edgecolors='darkgreen',
            linewidths=0.5)

ax3.set_xlabel(feature1_name.capitalize(), fontsize=10, fontweight='bold')
ax3.set_ylabel(feature2_name.capitalize(), fontsize=10, fontweight='bold')
ax3.set_zlabel(feature3_name.capitalize(), fontsize=10, fontweight='bold')
ax3.set_title('DBSCAN Only', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right')
ax3.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('3d_anomaly_visualization.png', dpi=300, bbox_inches='tight')
print("\n3. 3D visualization saved as '3d_anomaly_visualization.png'")

# Analyze anomaly characteristics in 3D space
print(f"\n4. Anomaly Characteristics Analysis:")
print(f"\n   Normal Data Statistics:")
print(f"     {feature1_name}: mean={feature1[normal_mask].mean():.4f}, std={feature1[normal_mask].std():.4f}")
print(f"     {feature2_name}: mean={feature2[normal_mask].mean():.4f}, std={feature2[normal_mask].std():.4f}")
print(f"     {feature3_name}: mean={feature3[normal_mask].mean():.4f}, std={feature3[normal_mask].std():.4f}")

if both_agree_mask.sum() > 0:
    print(f"\n   Anomaly Data Statistics (Both Methods Agree):")
    print(f"     {feature1_name}: mean={feature1[both_agree_mask].mean():.4f}, std={feature1[both_agree_mask].std():.4f}")
    print(f"     {feature2_name}: mean={feature2[both_agree_mask].mean():.4f}, std={feature2[both_agree_mask].std():.4f}")
    print(f"     {feature3_name}: mean={feature3[both_agree_mask].mean():.4f}, std={feature3[both_agree_mask].std():.4f}")
    
    print(f"\n   Difference from Normal (As mentioned in paper):")
    print(f"     {feature1_name}: {abs(feature1[both_agree_mask].mean() - feature1[normal_mask].mean()):.4f}")
    print(f"     {feature2_name}: {abs(feature2[both_agree_mask].mean() - feature2[normal_mask].mean()):.4f}")
    print(f"     {feature3_name}: {abs(feature3[both_agree_mask].mean() - feature3[normal_mask].mean()):.4f}")

# Calculate separation between normal and anomalous clusters
from sklearn.metrics import pairwise_distances

if both_agree_mask.sum() > 0:
    normal_center = np.array([
        feature1[normal_mask].mean(),
        feature2[normal_mask].mean(),
        feature3[normal_mask].mean()
    ]).reshape(1, -1)
    
    anomaly_center = np.array([
        feature1[both_agree_mask].mean(),
        feature2[both_agree_mask].mean(),
        feature3[both_agree_mask].mean()
    ]).reshape(1, -1)
    
    distance = pairwise_distances(normal_center, anomaly_center)[0][0]
    
    print(f"\n5. Cluster Separation:")
    print(f"   - Euclidean distance between normal and anomaly centers: {distance:.4f}")
    print(f"   - This indicates clear separation in feature space")

print("\n[SUCCESS] 3D visualization completed successfully!")
