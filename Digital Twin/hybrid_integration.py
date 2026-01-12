import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

print("=" * 60)
print("STEP 5: HYBRID INTEGRATION & WEIGHTED VOTING")
print("=" * 60)

# Load results from both algorithms
iso_results = pd.read_csv('isolation_forest_results.csv')
dbscan_results = pd.read_csv('dbscan_results.csv')

iso_labels = iso_results['iso_label'].values
iso_scores = iso_results['iso_score'].values
dbscan_labels = dbscan_results['dbscan_label'].values
dbscan_scores = dbscan_results['dbscan_score'].values

# Paper's Configuration
WEIGHT_ISO = 0.6  # Isolation Forest weight
WEIGHT_DBSCAN = 0.4  # DBSCAN weight
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to flag as anomaly

print(f"\n1. Weighted Voting Configuration:")
print(f" - Isolation Forest weight: {WEIGHT_ISO} ({WEIGHT_ISO*100}%)")
print(f" - DBSCAN weight: {WEIGHT_DBSCAN} ({WEIGHT_DBSCAN*100}%)")
print(f" - Confidence threshold: {CONFIDENCE_THRESHOLD}")

# Calculate weighted combined scores
hybrid_scores = (WEIGHT_ISO * iso_scores) + (WEIGHT_DBSCAN * dbscan_scores)

print(f"\n2. Hybrid Score Statistics:")
print(f" - Min score: {hybrid_scores.min():.4f}")
print(f" - Max score: {hybrid_scores.max():.4f}")
print(f" - Mean score: {hybrid_scores.mean():.4f}")
print(f" - Median score: {np.median(hybrid_scores):.4f}")
print(f" - Std deviation: {hybrid_scores.std():.4f}")

# Apply confidence threshold
hybrid_labels = np.where(hybrid_scores >= CONFIDENCE_THRESHOLD, 1, 0)

print(f"\n3. Hybrid Detection Results:")
print(f" - Total data points: {len(hybrid_labels)}")
print(f" - Anomalies detected: {hybrid_labels.sum()}")
print(f" - Anomaly rate: {hybrid_labels.sum()/len(hybrid_labels)*100:.2f}%")
print(f" - Normal points: {len(hybrid_labels) - hybrid_labels.sum()}")

# Agreement analysis
print(f"\n4. Algorithm Agreement Analysis:")
both_agree_anomaly = np.sum((iso_labels == 1) & (dbscan_labels == 1))
both_agree_normal = np.sum((iso_labels == 0) & (dbscan_labels == 0))
iso_only = np.sum((iso_labels == 1) & (dbscan_labels == 0))
dbscan_only = np.sum((iso_labels == 0) & (dbscan_labels == 1))
total_agreement = both_agree_anomaly + both_agree_normal
agreement_rate = total_agreement / len(iso_labels) * 100

print(f" - Both detect anomaly: {both_agree_anomaly}")
print(f" - Both detect normal: {both_agree_normal}")
print(f" - Only Isolation Forest: {iso_only}")
print(f" - Only DBSCAN: {dbscan_only}")
print(f" - Total agreement: {total_agreement} ({agreement_rate:.2f}%)")

# Comparison of methods
print(f"\n5. Detection Method Comparison:")
print(f" - Isolation Forest anomalies: {iso_labels.sum()}")
print(f" - DBSCAN anomalies: {dbscan_labels.sum()}")
print(f" - Hybrid method anomalies: {hybrid_labels.sum()}")

# Score correlation
pearson_corr, pearson_p = pearsonr(iso_scores, dbscan_scores)
spearman_corr, spearman_p = spearmanr(iso_scores, dbscan_scores)

print(f"\n6. Score Correlation Between Methods:")
print(f" - Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
print(f" - Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")

# Visualizations
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Score comparison scatter plot
ax1 = fig.add_subplot(gs[0, 0])
scatter = ax1.scatter(iso_scores, dbscan_scores, c=hybrid_labels,
                     cmap='coolwarm', alpha=0.5, s=10)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect agreement')
ax1.set_xlabel('Isolation Forest Score')
ax1.set_ylabel('DBSCAN Score')
ax1.set_title(f'Score Correlation (r={pearson_corr:.3f})')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Hybrid Label')

# 2. Venn diagram representation
ax2 = fig.add_subplot(gs[0, 1])
from matplotlib.patches import Circle
circle1 = Circle((0.3, 0.5), 0.3, alpha=0.3, color='blue', label='Isolation Forest')
circle2 = Circle((0.7, 0.5), 0.3, alpha=0.3, color='green', label='DBSCAN')
ax2.add_patch(circle1)
ax2.add_patch(circle2)
ax2.text(0.2, 0.5, str(iso_only), ha='center', va='center', fontsize=14, fontweight='bold')
ax2.text(0.5, 0.5, str(both_agree_anomaly), ha='center', va='center', fontsize=14, fontweight='bold')
ax2.text(0.8, 0.5, str(dbscan_only), ha='center', va='center', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.legend(loc='upper center')
ax2.set_title('Anomaly Detection Overlap')

# 3. Hybrid score distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(hybrid_scores, bins=50, alpha=0.7, color='purple', edgecolor='black')
ax3.axvline(CONFIDENCE_THRESHOLD, color='red', linestyle='--',
           linewidth=2, label=f'Threshold ({CONFIDENCE_THRESHOLD})')
ax3.set_xlabel('Hybrid Anomaly Score')
ax3.set_ylabel('Frequency')
ax3.set_title('Hybrid Score Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Time series comparison - All three methods
ax4 = fig.add_subplot(gs[1, :])
time_indices = np.arange(len(hybrid_scores))
ax4.scatter(time_indices, iso_scores, c='blue', alpha=0.3, s=1, label='Isolation Forest')
ax4.scatter(time_indices, dbscan_scores, c='green', alpha=0.3, s=1, label='DBSCAN')
ax4.scatter(time_indices, hybrid_scores, c='purple', alpha=0.5, s=2, label='Hybrid')
ax4.axhline(CONFIDENCE_THRESHOLD, color='red', linestyle='--', label='Threshold')
ax4.set_xlabel('Time Index')
ax4.set_ylabel('Anomaly Score')
ax4.set_title('Anomaly Scores Comparison Over Time')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Method agreement matrix
ax5 = fig.add_subplot(gs[2, 0])
agreement_matrix = np.array([
    [both_agree_normal, dbscan_only],
    [iso_only, both_agree_anomaly]
])
sns.heatmap(agreement_matrix, annot=True, fmt='d', cmap='Blues', ax=ax5,
           xticklabels=['DBSCAN: Normal', 'DBSCAN: Anomaly'],
           yticklabels=['ISO: Normal', 'ISO: Anomaly'])
ax5.set_title('Agreement Matrix')

# 6. Box plot comparison - FIXED: Use tick_labels instead of labels
ax6 = fig.add_subplot(gs[2, 1])
box_data = [
    iso_scores[iso_labels == 1],
    dbscan_scores[dbscan_labels == 1],
    hybrid_scores[hybrid_labels == 1]
]
bp = ax6.boxplot(box_data, tick_labels=['ISO', 'DBSCAN', 'Hybrid'], patch_artist=True)
colors = ['lightblue', 'lightgreen', 'plum']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax6.set_ylabel('Anomaly Score')
ax6.set_title('Anomaly Score Distribution by Method')
ax6.grid(True, alpha=0.3)

# 7. Cumulative detection comparison
ax7 = fig.add_subplot(gs[2, 2])
ax7.plot(np.cumsum(iso_labels), label='Isolation Forest', color='blue', linewidth=2)
ax7.plot(np.cumsum(dbscan_labels), label='DBSCAN', color='green', linewidth=2)
ax7.plot(np.cumsum(hybrid_labels), label='Hybrid', color='purple', linewidth=2)
ax7.set_xlabel('Time Index')
ax7.set_ylabel('Cumulative Anomaly Count')
ax7.set_title('Cumulative Anomaly Detection')
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.suptitle('Hybrid Anomaly Detection Analysis', fontsize=16, y=0.995)
plt.savefig('hybrid_integration_analysis.png', dpi=300, bbox_inches='tight')
print("\n7. Visualization saved as 'hybrid_integration_analysis.png'")

# Save hybrid results
hybrid_results_df = pd.DataFrame({
    'iso_label': iso_labels,
    'iso_score': iso_scores,
    'dbscan_label': dbscan_labels,
    'dbscan_score': dbscan_scores,
    'hybrid_score': hybrid_scores,
    'hybrid_label': hybrid_labels
})
hybrid_results_df.to_csv('hybrid_results.csv', index=False)
print("8. Hybrid results saved as 'hybrid_results.csv'")

# Identify high-confidence anomalies (both methods agree)
high_confidence_anomalies = np.where(
    (iso_labels == 1) & (dbscan_labels == 1) & (hybrid_scores >= 0.7)
)[0]

print(f"\n9. High-Confidence Anomalies (both methods agree + score >= 0.7):")
print(f" - Count: {len(high_confidence_anomalies)}")
if len(high_confidence_anomalies) > 0:
    print(f" - Top 10 indices: {high_confidence_anomalies[:10]}")
    print(f" - Average score: {hybrid_scores[high_confidence_anomalies].mean():.4f}")

# Summary statistics table
print("\n10. Summary Statistics Table:")
print("=" * 70)
print(f"{'Metric':<30} {'ISO Forest':<15} {'DBSCAN':<15} {'Hybrid':<15}")
print("=" * 70)
print(f"{'Anomalies Detected':<30} {iso_labels.sum():<15} {dbscan_labels.sum():<15} {hybrid_labels.sum():<15}")
print(f"{'Anomaly Rate (%)':<30} {iso_labels.sum()/len(iso_labels)*100:<15.2f} {dbscan_labels.sum()/len(dbscan_labels)*100:<15.2f} {hybrid_labels.sum()/len(hybrid_labels)*100:<15.2f}")
print(f"{'Mean Anomaly Score':<30} {iso_scores[iso_labels==1].mean():<15.4f} {dbscan_scores[dbscan_labels==1].mean():<15.4f} {hybrid_scores[hybrid_labels==1].mean():<15.4f}")
print(f"{'Max Score':<30} {iso_scores.max():<15.4f} {dbscan_scores.max():<15.4f} {hybrid_scores.max():<15.4f}")
print("=" * 70)

print("\n[SUCCESS] Hybrid integration completed successfully!")
