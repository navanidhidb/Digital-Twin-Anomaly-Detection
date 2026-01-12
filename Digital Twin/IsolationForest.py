import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

print("=" * 60)
print("STEP 3: ISOLATION FOREST ANOMALY DETECTION")
print("=" * 60)

# Load preprocessed data
preprocessed_data = pd.read_csv('data_preprocessed.csv').values

# Paper's Configuration
CONTAMINATION = 0.05  # 5% expected anomalies (range: 0.01-0.1)
N_ESTIMATORS = 200    # Number of trees
RANDOM_STATE = 42     # For reproducibility

print(f"\n1. Isolation Forest Configuration:")
print(f"   - Contamination: {CONTAMINATION} ({CONTAMINATION*100}%)")
print(f"   - Number of trees (n_estimators): {N_ESTIMATORS}")
print(f"   - Random state: {RANDOM_STATE}")

# Initialize Isolation Forest
print("\n2. Training Isolation Forest...")
iso_forest = IsolationForest(
    contamination=CONTAMINATION,
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    max_samples='auto',
    bootstrap=False,
    n_jobs=-1,  # Use all CPU cores
    verbose=0
)

# Fit and predict
iso_forest.fit(preprocessed_data)
print("   Training completed!")

# Get predictions: 1 = normal, -1 = anomaly
iso_predictions = iso_forest.predict(preprocessed_data)

# Convert to binary: 0 = normal, 1 = anomaly
iso_labels = np.where(iso_predictions == -1, 1, 0)

# Get anomaly scores (more negative = more anomalous)
iso_scores_raw = iso_forest.decision_function(preprocessed_data)

# Convert to confidence scores (0-1 range, higher = more anomalous)
# Normalize using min-max scaling
iso_scores = (iso_scores_raw - iso_scores_raw.min()) / (iso_scores_raw.max() - iso_scores_raw.min())
iso_scores = 1 - iso_scores  # Invert so higher = more anomalous

print(f"\n3. Isolation Forest Results:")
print(f"   - Total data points: {len(iso_labels)}")
print(f"   - Anomalies detected: {iso_labels.sum()}")
print(f"   - Anomaly rate: {iso_labels.sum()/len(iso_labels)*100:.2f}%")
print(f"   - Normal points: {len(iso_labels) - iso_labels.sum()}")

print(f"\n4. Anomaly Score Statistics:")
print(f"   - Min score: {iso_scores.min():.4f}")
print(f"   - Max score: {iso_scores.max():.4f}")
print(f"   - Mean score: {iso_scores.mean():.4f}")
print(f"   - Median score: {np.median(iso_scores):.4f}")

# Analyze score distribution
print(f"\n5. Score Distribution for Anomalies:")
anomaly_scores = iso_scores[iso_labels == 1]
print(f"   - Mean anomaly score: {anomaly_scores.mean():.4f}")
print(f"   - Min anomaly score: {anomaly_scores.min():.4f}")
print(f"   - Max anomaly score: {anomaly_scores.max():.4f}")

print(f"\n6. Score Distribution for Normal Points:")
normal_scores = iso_scores[iso_labels == 0]
print(f"   - Mean normal score: {normal_scores.mean():.4f}")
print(f"   - Max normal score: {normal_scores.max():.4f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Isolation Forest Analysis', fontsize=16)

# 1. Score distribution
axes[0, 0].hist(iso_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].axvline(iso_scores[iso_labels == 1].min(), color='red', 
                   linestyle='--', label=f'Anomaly threshold')
axes[0, 0].set_xlabel('Anomaly Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Anomaly Score Distribution')
axes[0, 0].legend()

# 2. Anomaly timeline
axes[0, 1].scatter(range(len(iso_scores)), iso_scores, 
                   c=iso_labels, cmap='coolwarm', alpha=0.6, s=1)
axes[0, 1].set_xlabel('Time Index')
axes[0, 1].set_ylabel('Anomaly Score')
axes[0, 1].set_title('Anomaly Scores Over Time')
axes[0, 1].axhline(y=0.6, color='red', linestyle='--', label='Threshold (0.6)')
axes[0, 1].legend()

# 3. Box plot comparison
box_data = [normal_scores, anomaly_scores]
axes[1, 0].boxplot(box_data, tick_labels=['Normal', 'Anomaly'])
axes[1, 0].set_ylabel('Anomaly Score')
axes[1, 0].set_title('Score Distribution: Normal vs Anomaly')
axes[1, 0].grid(True, alpha=0.3)

# 4. Cumulative anomaly count
cumulative_anomalies = np.cumsum(iso_labels)
axes[1, 1].plot(cumulative_anomalies, color='red', linewidth=2)
axes[1, 1].set_xlabel('Time Index')
axes[1, 1].set_ylabel('Cumulative Anomaly Count')
axes[1, 1].set_title('Cumulative Anomalies Detected')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('isolation_forest_analysis.png', dpi=300, bbox_inches='tight')
print("\n7. Visualization saved as 'isolation_forest_analysis.png'")

# Save results
results_df = pd.DataFrame({
    'iso_label': iso_labels,
    'iso_score': iso_scores
})
results_df.to_csv('isolation_forest_results.csv', index=False)
print("8. Results saved as 'isolation_forest_results.csv'")

# Show top anomalies
print("\n9. Top 10 Anomalies by Isolation Forest:")
top_anomalies = np.argsort(iso_scores)[-10:][::-1]
for rank, idx in enumerate(top_anomalies, 1):
    print(f"   Rank {rank}: Index {idx}, Score: {iso_scores[idx]:.4f}")