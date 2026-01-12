import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

print("=" * 60)
print("DBSCAN ANOMALY DETECTION")
print("=" * 60)

# Load preprocessed data
preprocessed_data = pd.read_csv('data_preprocessed.csv').values
print(f"\nData loaded: {preprocessed_data.shape[0]} samples, {preprocessed_data.shape[1]} features")
print(f"Data range: [{preprocessed_data.min():.4f}, {preprocessed_data.max():.4f}]")

# ============================================================================
# STEP 1: K-DISTANCE ANALYSIS (Find optimal epsilon)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 1: K-Distance Analysis")
print("=" * 60)

k = 5  # Standard choice for DBSCAN
neighbors = NearestNeighbors(n_neighbors=k+1)
neighbors_fit = neighbors.fit(preprocessed_data)
distances, _ = neighbors_fit.kneighbors(preprocessed_data)
distances_sorted = np.sort(distances[:, k], axis=0)

# Key percentiles
p90 = np.percentile(distances_sorted, 90)
p95 = np.percentile(distances_sorted, 95)
p99 = np.percentile(distances_sorted, 99)

print(f"K-distance statistics (k={k}):")
print(f"  90th percentile: {p90:.4f}")
print(f"  95th percentile: {p95:.4f}")
print(f"  99th percentile: {p99:.4f}")
print(f"\nRecommended epsilon range: {p90:.2f} - {p95:.2f}")

# Plot k-distance graph
plt.figure(figsize=(10, 6))
plt.plot(distances_sorted, linewidth=1.5, color='blue')
plt.axhline(y=p95, color='red', linestyle='--', linewidth=2, label=f'95th percentile: {p95:.3f}')
plt.axhline(y=p90, color='orange', linestyle='--', linewidth=2, label=f'90th percentile: {p90:.3f}')
plt.xlabel('Points (sorted by distance)', fontsize=12)
plt.ylabel(f'{k}-Nearest Neighbor Distance', fontsize=12)
plt.title('K-Distance Graph (Elbow Method for Epsilon Selection)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('k_distance_graph.png', dpi=300, bbox_inches='tight')
print("[OK] K-distance graph saved as 'k_distance_graph.png'")

# ============================================================================
# STEP 2: GRID SEARCH (Simplified - 18 combinations)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Parameter Grid Search")
print("=" * 60)

# Focused parameter ranges based on k-distance
EPS_VALUES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
MIN_SAMPLES_VALUES = [5, 10, 15]

print(f"Testing {len(EPS_VALUES)} epsilon x {len(MIN_SAMPLES_VALUES)} min_samples = {len(EPS_VALUES) * len(MIN_SAMPLES_VALUES)} combinations")
print(f"Epsilon range: {min(EPS_VALUES)} to {max(EPS_VALUES)}")
print(f"Min_samples range: {MIN_SAMPLES_VALUES}")

print("\n" + "-" * 80)
print(f"{'Eps':>6} {'MinS':>6} {'Clusters':>8} {'Anomalies':>10} {'Rate %':>8} {'Silhouette':>12}")
print("-" * 80)

results_list = []
best_config = None
best_score = -999

for eps in EPS_VALUES:
    for min_samples in MIN_SAMPLES_VALUES:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(preprocessed_data)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_rate = n_noise / len(labels) * 100
        
        # Calculate silhouette score
        silhouette = -1
        if n_clusters >= 2 and n_noise < len(labels):
            mask = labels != -1
            if mask.sum() > 1:
                try:
                    silhouette = silhouette_score(preprocessed_data[mask], labels[mask])
                except:
                    silhouette = -1
        
        result = {
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_rate': noise_rate,
            'silhouette': silhouette,
            'labels': labels
        }
        results_list.append(result)
        
        # Select best: prefer 3-10% anomaly rate with good silhouette
        if 3 <= noise_rate <= 10:
            score = silhouette if silhouette > 0 else -999
            if score > best_score:
                best_score = score
                best_config = result
        
        marker = "*" if (3 <= noise_rate <= 10 and silhouette > 0) else " "
        print(f"{marker} {eps:5.2f} {min_samples:6d} {n_clusters:8d} {n_noise:10d} {noise_rate:7.2f}% {silhouette:11.4f}")

print("-" * 80)

# If no ideal config found, pick one closest to 5% anomaly rate
if best_config is None:
    configs_with_anomalies = [r for r in results_list if r['n_noise'] > 0]
    if configs_with_anomalies:
        best_config = min(configs_with_anomalies, key=lambda x: abs(x['noise_rate'] - 5.0))
        print("\n[WARNING] No configuration in ideal range (3-10%). Using closest to 5% anomaly rate.")
    else:
        print("\n[ERROR] NO ANOMALIES DETECTED! Data may be too uniform or epsilon too large.")
        print("  Try smaller epsilon values or check data preprocessing.")
        exit(1)

# ============================================================================
# STEP 3: FINAL RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: Selected Configuration")
print("=" * 60)

best_eps = best_config['eps']
best_min_samples = best_config['min_samples']
best_labels = best_config['labels']
best_n_clusters = best_config['n_clusters']
best_n_noise = best_config['n_noise']
best_noise_rate = best_config['noise_rate']
best_silhouette = best_config['silhouette']

print(f"Epsilon (eps): {best_eps}")
print(f"Min_samples: {best_min_samples}")
print(f"Clusters found: {best_n_clusters}")
print(f"Anomalies detected: {best_n_noise}")
print(f"Anomaly rate: {best_noise_rate:.2f}%")
print(f"Silhouette score: {best_silhouette:.4f}")

# Convert to binary labels
dbscan_labels = np.where(best_labels == -1, 1, 0)

# Calculate confidence scores
print("\nCalculating anomaly confidence scores...")

cluster_centers = []
unique_clusters = [c for c in set(best_labels) if c != -1]

for cluster_id in unique_clusters:
    cluster_points = preprocessed_data[best_labels == cluster_id]
    if len(cluster_points) > 0:
        center = cluster_points.mean(axis=0)
        cluster_centers.append(center)

if len(cluster_centers) > 0:
    cluster_centers = np.array(cluster_centers)
    distances_to_clusters = pairwise_distances(preprocessed_data, cluster_centers)
    min_distances = distances_to_clusters.min(axis=1)
    
    if min_distances.max() > min_distances.min():
        dbscan_scores = (min_distances - min_distances.min()) / (min_distances.max() - min_distances.min())
    else:
        dbscan_scores = np.where(dbscan_labels == 1, 0.9, 0.1)
    
    # Ensure anomalies have high scores
    anomaly_mask = dbscan_labels == 1
    if anomaly_mask.any():
        dbscan_scores[anomaly_mask] = np.maximum(dbscan_scores[anomaly_mask], 0.6)
else:
    dbscan_scores = np.where(dbscan_labels == 1, 0.9, 0.1)

print(f"Score range: [{dbscan_scores.min():.4f}, {dbscan_scores.max():.4f}]")
if dbscan_labels.sum() > 0:
    print(f"Mean anomaly score: {dbscan_scores[dbscan_labels==1].mean():.4f}")
    print(f"Mean normal score: {dbscan_scores[dbscan_labels==0].mean():.4f}")

# ============================================================================
# STEP 4: VISUALIZATIONS (Key plots only)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Creating Visualizations")
print("=" * 60)

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. Anomaly scores timeline
ax1 = fig.add_subplot(gs[0, :2])
scatter = ax1.scatter(range(len(dbscan_scores)), dbscan_scores,
                     c=dbscan_labels, cmap='RdYlGn_r', alpha=0.6, s=3)
ax1.set_xlabel('Time Index', fontsize=11)
ax1.set_ylabel('Anomaly Score', fontsize=11)
ax1.set_title('Anomaly Scores Over Time', fontsize=13, fontweight='bold')
ax1.axhline(y=0.6, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Anomaly (1=Yes)')

# 2. Cluster distribution
ax2 = fig.add_subplot(gs[0, 2])
unique, counts = np.unique(best_labels, return_counts=True)
colors = ['red' if u == -1 else 'skyblue' for u in unique]
bars = ax2.bar([f'{u}' for u in unique], counts, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Cluster ID', fontsize=11)
ax2.set_ylabel('Count (log scale)', fontsize=11)
ax2.set_title(f'Cluster Distribution\n(eps={best_eps}, min_s={best_min_samples})', fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')

# Highlight anomaly bar
for bar, u in zip(bars, unique):
    if u == -1:
        bar.set_linewidth(3)

# 3. Score distribution
ax3 = fig.add_subplot(gs[1, 0])
if dbscan_labels.sum() > 0:
    ax3.hist(dbscan_scores[dbscan_labels==1], bins=40, alpha=0.7, color='red',
            edgecolor='black', label=f'Anomalies (n={dbscan_labels.sum()})', density=True)
if (dbscan_labels==0).sum() > 0:
    ax3.hist(dbscan_scores[dbscan_labels==0], bins=40, alpha=0.7, color='green',
            edgecolor='black', label=f'Normal (n={(dbscan_labels==0).sum()})', density=True)
ax3.set_xlabel('Anomaly Score', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title('Score Distribution', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Detected anomalies timeline
ax4 = fig.add_subplot(gs[1, 1])
if dbscan_labels.sum() > 0:
    anomaly_indices = np.where(dbscan_labels == 1)[0]
    scatter2 = ax4.scatter(anomaly_indices, dbscan_scores[anomaly_indices],
                          c=dbscan_scores[anomaly_indices], cmap='Reds', s=40, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Time Index', fontsize=11)
    ax4.set_ylabel('Anomaly Score', fontsize=11)
    ax4.set_title(f'Detected Anomalies (n={len(anomaly_indices)})', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax4, label='Score')
else:
    ax4.text(0.5, 0.5, 'No Anomalies Detected',
            ha='center', va='center', transform=ax4.transAxes, fontsize=12, color='red')

# 5. Cumulative anomalies
ax5 = fig.add_subplot(gs[1, 2])
cumulative = np.cumsum(dbscan_labels)
ax5.plot(cumulative, color='darkred', linewidth=2)
ax5.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='red')
ax5.set_xlabel('Time Index', fontsize=11)
ax5.set_ylabel('Cumulative Count', fontsize=11)
ax5.set_title(f'Cumulative Anomalies\n(Total: {dbscan_labels.sum()})', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

plt.suptitle(f'DBSCAN Anomaly Detection Results', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('dbscan_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Analysis visualization saved as 'dbscan_analysis.png'")

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: Saving Results")
print("=" * 60)

# Save main results
results_df = pd.DataFrame({
    'dbscan_label': dbscan_labels,
    'dbscan_score': dbscan_scores,
    'cluster_id': best_labels
})
results_df.to_csv('dbscan_results.csv', index=False)
print("[OK] Results saved as 'dbscan_results.csv'")

# Save parameter search results
pd.DataFrame(results_list).to_csv('dbscan_parameter_search.csv', index=False)
print("[OK] Parameter search saved as 'dbscan_parameter_search.csv'")

# Top anomalies report
if dbscan_labels.sum() > 0:
    print("\n" + "=" * 60)
    print("Top 15 Anomalies (Highest Confidence)")
    print("=" * 60)
    print(f"{'Rank':>6} {'Index':>8} {'Score':>10} {'Cluster':>10}")
    print("-" * 60)
    
    top_indices = np.argsort(dbscan_scores)[-20:][::-1]
    count = 0
    for idx in top_indices:
        if dbscan_labels[idx] == 1:
            count += 1
            cluster = best_labels[idx]
            print(f"{count:6d} {idx:8d} {dbscan_scores[idx]:10.4f} {cluster:10d}")
            if count >= 15:
                break

print("\n" + "=" * 60)
print(f"[SUCCESS] COMPLETE: {dbscan_labels.sum()} anomalies detected ({best_noise_rate:.2f}%)")
print("=" * 60)
