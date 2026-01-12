import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("=" * 60)
print("STEP 8: FEATURE CONTRIBUTION ANALYSIS (Z-SCORE)")
print("=" * 60)

# Load data
df_original = pd.read_csv('dataset_final.csv')
df_preprocessed = pd.read_csv('data_preprocessed.csv')
hybrid_results = pd.read_csv('hybrid_results.csv')

# Get labels
hybrid_labels = hybrid_results['hybrid_label'].values

# Use original (non-scaled) data for Z-score calculation
sensor_cols = df_preprocessed.columns.tolist()

# Ensure length matches
min_len = min(len(df_original), len(hybrid_labels))

# Extract sensor data from original dataset
sensor_data_dict = {}
for col in sensor_cols:
    # Try to find matching column in original data
    matching_cols = [c for c in df_original.columns if col.lower() in c.lower()]
    if matching_cols:
        sensor_data_dict[col] = df_original[matching_cols[0]].values[:min_len]
    else:
        # Use preprocessed data if original not found
        sensor_data_dict[col] = df_preprocessed[col].values[:min_len]

sensor_data = pd.DataFrame(sensor_data_dict)
hybrid_labels = hybrid_labels[:min_len]

print(f"\n1. Analyzing {len(sensor_cols)} sensor features:")
print(f"   Features: {sensor_cols}")

# Calculate Z-scores for each feature
print(f"\n2. Calculating Z-scores...")

# Z-score formula: (value - mean) / std_dev
feature_zscores = {}
feature_contributions = {}

for col in sensor_cols:
    # Calculate mean and std from normal data only
    normal_data = sensor_data[col].values[hybrid_labels == 0]
    mean_normal = normal_data.mean()
    std_normal = normal_data.std()
    
    # Calculate Z-scores for all data
    zscores = (sensor_data[col].values - mean_normal) / std_normal
    feature_zscores[col] = zscores
    
    # Calculate mean absolute Z-score for anomalies
    anomaly_zscores = np.abs(zscores[hybrid_labels == 1])
    mean_anomaly_zscore = anomaly_zscores.mean() if len(anomaly_zscores) > 0 else 0
    feature_contributions[col] = mean_anomaly_zscore
    
    print(f"\n   {col.capitalize()}:")
    print(f"     Normal data: mean={mean_normal:.2f}, std={std_normal:.2f}")
    if len(anomaly_zscores) > 0:
        print(f"     Anomaly mean Z-score: {mean_anomaly_zscore:.4f}")
        print(f"     Max anomaly Z-score: {anomaly_zscores.max():.4f}")
        print(f"     % anomalies with |Z| > 2: {(anomaly_zscores > 2).sum() / len(anomaly_zscores) * 100:.1f}%")
        print(f"     % anomalies with |Z| > 3: {(anomaly_zscores > 3).sum() / len(anomaly_zscores) * 100:.1f}%")

# Sort features by contribution
sorted_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)

print(f"\n3. Feature Contribution Ranking (by mean Z-score):")
for rank, (feature, zscore) in enumerate(sorted_features, 1):
    print(f"   {rank}. {feature.capitalize()}: {zscore:.4f}")

# Figure 4 from paper: Features Contributing Most to Anomalies
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Feature Contribution Analysis\n(Similar to Figure 4 in Paper)',
             fontsize=16, fontweight='bold')

# Plot 1: Bar chart of mean Z-scores (MAIN PLOT FROM PAPER)
ax1 = axes[0, 0]
features_list = [f[0] for f in sorted_features]
zscores_list = [f[1] for f in sorted_features]
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(features_list)))

bars = ax1.barh(features_list, zscores_list, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Mean Absolute Z-Score', fontsize=12, fontweight='bold')
ax1.set_ylabel('Sensor Feature', fontsize=12, fontweight='bold')
ax1.set_title('Features Contributing Most to Anomalies\n(Higher Z-score = More Important)',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, zscores_list)):
    ax1.text(val + 0.05, i, f'{val:.3f}',
             va='center', fontweight='bold', fontsize=10)

# Plot 2: Z-score distribution for each feature - FIXED: Use tick_labels
ax2 = axes[0, 1]
box_data = []
labels_list = []

for col in sorted_features:
    feature_name = col[0]
    anomaly_zscores = np.abs(feature_zscores[feature_name][hybrid_labels == 1])
    if len(anomaly_zscores) > 0:
        box_data.append(anomaly_zscores)
        labels_list.append(feature_name)

bp = ax2.boxplot(box_data, tick_labels=labels_list, patch_artist=True, vert=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax2.set_ylabel('Absolute Z-Score', fontsize=11, fontweight='bold')
ax2.set_xlabel('Sensor Feature', fontsize=11, fontweight='bold')
ax2.set_title('Z-Score Distribution for Anomalies', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Heatmap of Z-scores over time for anomalies
ax3 = axes[1, 0]
anomaly_indices = np.where(hybrid_labels == 1)[0][:100]  # Show first 100 anomalies

if len(anomaly_indices) > 0:
    zscore_matrix = np.array([feature_zscores[col][anomaly_indices] for col in sensor_cols])
    im = ax3.imshow(zscore_matrix, aspect='auto', cmap='RdYlBu_r',
                    vmin=-5, vmax=5, interpolation='nearest')
    ax3.set_yticks(range(len(sensor_cols)))
    ax3.set_yticklabels([col.capitalize() for col in sensor_cols])
    ax3.set_xlabel('Anomaly Sample Index', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Sensor Feature', fontsize=11, fontweight='bold')
    ax3.set_title('Z-Score Heatmap for Anomalies', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Z-Score')

# Plot 4: Cumulative contribution
ax4 = axes[1, 1]
cumulative_contribution = np.cumsum([f[1] for f in sorted_features])
total_contribution = cumulative_contribution[-1]
cumulative_percentage = (cumulative_contribution / total_contribution) * 100

ax4.plot(range(1, len(sorted_features) + 1), cumulative_percentage,
         marker='o', linewidth=2, markersize=8, color='darkred')
ax4.axhline(y=80, color='green', linestyle='--', linewidth=2, label='80% threshold')
ax4.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
ax4.set_ylabel('Cumulative Contribution (%)', fontsize=11, fontweight='bold')
ax4.set_title('Cumulative Feature Contribution', fontsize=12, fontweight='bold')
ax4.set_xticks(range(1, len(sorted_features) + 1))
ax4.set_xticklabels([f[0] for f in sorted_features], rotation=45)
ax4.grid(True, alpha=0.3)
ax4.legend()

# Add text annotation for 80% threshold
features_for_80 = np.where(cumulative_percentage >= 80)[0]
if len(features_for_80) > 0:
    first_80_idx = features_for_80[0]
    ax4.annotate(f'{first_80_idx + 1} features\nexplain 80%',
                 xy=(first_80_idx + 1, 80),
                 xytext=(first_80_idx + 1, 60),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2),
                 fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_contribution_analysis.png', dpi=300, bbox_inches='tight')
print("\n4. Visualization saved as 'feature_contribution_analysis.png'")

# Statistical significance testing
print(f"\n5. Statistical Significance Testing:")
print(f"   (Comparing normal vs anomaly distributions)")

for col in sensor_cols:
    normal_values = sensor_data[col].values[hybrid_labels == 0]
    anomaly_values = sensor_data[col].values[hybrid_labels == 1]
    
    if len(anomaly_values) > 0:
        # T-test
        t_stat, p_value = stats.ttest_ind(normal_values, anomaly_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(normal_values)-1)*normal_values.std()**2 + 
                              (len(anomaly_values)-1)*anomaly_values.std()**2) / 
                             (len(normal_values) + len(anomaly_values) - 2))
        cohens_d = (anomaly_values.mean() - normal_values.mean()) / pooled_std
        
        print(f"\n   {col.capitalize()}:")
        print(f"     T-statistic: {t_stat:.4f}")
        print(f"     P-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        print(f"     Cohen's d: {abs(cohens_d):.4f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")

# Save detailed Z-score results
zscore_results = pd.DataFrame(feature_zscores)
zscore_results['is_anomaly'] = hybrid_labels
zscore_results.to_csv('feature_zscores.csv', index=False)
print(f"\n6. Z-score results saved as 'feature_zscores.csv'")

# Summary interpretation (as in paper)
print(f"\n7. Key Findings (Similar to Paper's Conclusion):")
print(f"   - Most dominant feature: {sorted_features[0][0].upper()} (Z-score: {sorted_features[0][1]:.4f})")
print(f"   - This feature shows the highest deviation from normal behavior during anomalies")
print(f"   - Consistent with temperature spikes and sudden shifts observed in Figure 2")

if len(sorted_features) > 1:
    print(f"   - Secondary contributors: {sorted_features[1][0]} and {sorted_features[2][0] if len(sorted_features) > 2 else 'N/A'}")
    print(f"   - These provide complementary information for detection")

print(f"\n   - The combination of multiple features enables robust anomaly detection")
print(f"   - Higher Z-scores indicate stronger evidence of anomalous behavior")

print("\n[SUCCESS] Feature contribution analysis completed successfully!")
