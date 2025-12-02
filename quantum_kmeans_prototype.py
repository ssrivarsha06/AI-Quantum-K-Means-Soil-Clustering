#!/usr/bin/env python3
"""
quantum_kmeans_prototype.py

Hybrid Quantum-KMeans prototype (emulation + optional Qiskit swap-test).
Save next to your dataset zip file and run:
    python quantum_kmeans_prototype.py --data croprecommondation.zip --k 3
"""

import os
import zipfile
import argparse
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -------- helpers to load CSV from zip --------
def load_csv_from_zip(zip_path):
    """Open a zip that contains a single CSV and return a DataFrame."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} not found")
    with zipfile.ZipFile(zip_path, 'r') as z:
        # try to find the first CSV inside
        for name in z.namelist():
            if name.lower().endswith('.csv'):
                return pd.read_csv(z.open(name))
    raise ValueError("No CSV file found inside the zip")

# -------- preprocessing --------
def prepare_features(df, feature_cols):
    """Select feature columns, encode categorical, scale and normalize."""
    df_copy = df[feature_cols].copy()
    # Encode categorical columns (object or string types)
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
    X = df_copy.astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_norm = normalize(X_scaled, norm='l2', axis=1)  # amplitude-style normalization per sample
    return X_scaled, X_norm

# -------- quantum-style KMeans emulation with metrics --------
def quantum_kmeans_emulation(X_raw, X_norm, k=3, max_iter=200, random_state=42):
    """
    Hybrid loop with convergence tracking:
      - use inner product between normalized sample and centroid as 'similarity'
      - centroid update: classical mean of raw features, then L2-normalize to get new amplitude
    """
    start_time = time.time()
    np.random.seed(random_state)
    n_samples, n_features = X_norm.shape
    idx = np.random.choice(n_samples, k, replace=False)
    centroids_raw = X_raw[idx].astype(float)
    centroids = normalize(centroids_raw, norm='l2', axis=1)
    
    labels = np.zeros(n_samples, dtype=int)
    convergence_history = []
    inertia_history = []
    
    for it in range(max_iter):
        sims = X_norm @ centroids.T  # similarities (n_samples x k)
        new_labels = np.argmax(sims, axis=1)
        
        # Calculate inertia (sum of squared distances to centroids)
        inertia = 0
        for j in range(k):
            cluster_members = X_raw[new_labels == j]
            if len(cluster_members) > 0:
                centroid_raw = centroids_raw[j]
                inertia += np.sum((cluster_members - centroid_raw) ** 2)
        inertia_history.append(inertia)
        
        new_centroids_raw = np.zeros_like(centroids_raw)
        for j in range(k):
            members = X_raw[new_labels==j]
            if len(members) == 0:
                new_centroids_raw[j] = X_raw[np.random.choice(n_samples)]
            else:
                new_centroids_raw[j] = members.mean(axis=0)
        new_centroids = normalize(new_centroids_raw, norm='l2', axis=1)
        
        # Check convergence
        if np.array_equal(new_labels, labels):
            print(f"[emulation] Converged at iter {it}")
            labels = new_labels
            centroids = new_centroids
            centroids_raw = new_centroids_raw
            break
        
        labels = new_labels
        centroids = new_centroids
        centroids_raw = new_centroids_raw
        convergence_history.append(it)
    else:
        print("[emulation] Reached max iterations without full convergence")
    
    runtime = time.time() - start_time
    final_inertia = inertia_history[-1] if inertia_history else 0
    
    return labels, centroids, {
        'convergence_iterations': len(convergence_history),
        'runtime': runtime,
        'inertia': final_inertia,
        'inertia_history': inertia_history
    }

# -------- Classical KMeans with metrics --------
def classical_kmeans_with_metrics(X_raw, k=3, random_state=42):
    """Run classical KMeans and track metrics."""
    start_time = time.time()
    km = KMeans(n_clusters=k, random_state=random_state, max_iter=200)
    labels = km.fit_predict(X_raw)
    runtime = time.time() - start_time
    
    return labels, km, {
        'convergence_iterations': km.n_iter_,
        'runtime': runtime,
        'inertia': km.inertia_
    }

# -------- Metrics calculation --------
def calculate_clustering_metrics(X, labels_quantum, labels_classical):
    """Calculate and compare clustering quality metrics."""
    metrics = {}
    
    # Silhouette Score (higher is better)
    if len(np.unique(labels_quantum)) > 1:
        metrics['silhouette_quantum'] = silhouette_score(X, labels_quantum)
    else:
        metrics['silhouette_quantum'] = -1
        
    if len(np.unique(labels_classical)) > 1:
        metrics['silhouette_classical'] = silhouette_score(X, labels_classical)
    else:
        metrics['silhouette_classical'] = -1
    
    # Adjusted Rand Index (agreement between quantum and classical)
    metrics['adjusted_rand_index'] = adjusted_rand_score(labels_quantum, labels_classical)
    
    return metrics

# -------- Optional Qiskit swap test (very small scale) --------
def try_import_qiskit():
    try:
        import qiskit
        from qiskit import QuantumCircuit
        from qiskit_aer import Aer
        from qiskit import execute
        return (qiskit, QuantumCircuit, Aer, execute)
    except ImportError:
        try:
            # Try alternative import for newer Qiskit versions
            import qiskit
            from qiskit import QuantumCircuit
            from qiskit import BasicAer as Aer
            from qiskit import execute
            return (qiskit, QuantumCircuit, Aer, execute)
        except ImportError:
            print("Qiskit not available or failed to import. To run swap-test, install qiskit.")
            return (None, None, None, None)

def pad_to_pow2(v):
    """Pad vector v to the nearest power of two length with zeros (then normalize)."""
    L = len(v)
    pow2 = 1
    while pow2 < L:
        pow2 <<= 1
    if pow2 == L:
        vec = v.astype(np.complex128)
    else:
        vec = np.zeros(pow2, dtype=np.complex128)
        vec[:L] = v.astype(np.complex128)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def swap_test_qiskit(vec1, vec2, shots=1024):
    """
    Use Qiskit Swap Test to approximate |<psi|phi>|^2 (fidelity).
    NOTE: This requires Qiskit installed. This runs on a local simulator (Aer) if available.
    """
    qiskit, QuantumCircuit, Aer, execute = try_import_qiskit()
    if qiskit is None:
        raise ImportError("Qiskit not available")

    # pad both to same power-of-two dimension
    v1 = pad_to_pow2(vec1)
    v2 = pad_to_pow2(vec2)
    if len(v1) != len(v2):
        raise ValueError("Vectors padded to different sizes")

    n_state_qubits = int(np.log2(len(v1)))
    total_qubits = 1 + 2 * n_state_qubits  # ancilla + two registers
    qc = QuantumCircuit(total_qubits, 1)

    # initialize registers (ancilla is qubit 0)
    # registers: ancilla=0, reg1=1..n, reg2=n+1..2n
    reg1 = list(range(1, 1 + n_state_qubits))
    reg2 = list(range(1 + n_state_qubits, 1 + 2 * n_state_qubits))

    qc.h(0)  # ancilla in superposition
    # initialize states (Qiskit's initialize expects full amplitude vector)
    qc.initialize(v1.tolist(), reg1)
    qc.initialize(v2.tolist(), reg2)
    # controlled swap for each pair
    for q1, q2 in zip(reg1, reg2):
        qc.cswap(0, q1, q2)
    qc.h(0)
    qc.measure(0, 0)

    backend = None
    try:
        backend = Aer.get_backend('qasm_simulator')
    except Exception:
        # fallback to basic simulator if Aer isn't available
        from qiskit import BasicAer
        backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc, backend=backend, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)
    p0 = counts.get('0', 0) / shots
    # swap test formula: p0 = (1 + |<psi|phi>|^2) / 2
    fidelity = max(0.0, 2 * p0 - 1)
    return fidelity

# -------- plotting & saving --------
def plot_clusters_and_save(X_raw, labels_em, labels_classical, out_png='/tmp/cluster_compare.png'):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X_raw)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.scatter(X2[:,0], X2[:,1], c=labels_em, s=8)
    plt.title('Quantum-emulation clustering')
    plt.subplot(1,2,2)
    plt.scatter(X2[:,0], X2[:,1], c=labels_classical, s=8)
    plt.title('Classical KMeans clustering')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved visualization to {out_png}")

# -------- main CLI --------
def main(args):
    df = load_csv_from_zip(args.data)
    print("Loaded data with shape:", df.shape)
    
    # choose sensible defaults depending on dataset columns
    if set(['N','P','K','ph','temperature','humidity','rainfall']).issubset(df.columns):
        features = ['N','P','K','ph','temperature','humidity','rainfall']
    elif set(['pH','Nitrogen','Phosphorus','Potassium']).issubset(df.columns):
        # nutrients dataset naming
        cand = df.columns.str.lower()
        chosen = []
        for want in ['ph','nitrogen','phosphorus','potassium','temperature','rainfall']:
            for col in df.columns:
                if col.lower().startswith(want):
                    chosen.append(col)
                    break
        features = chosen[:6]
    else:
        # fallback: take first 5 numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = num_cols[:6]
    print("Selected features:", features)

    X_raw, X_norm = prepare_features(df, features)
    print("Prepared features shape:", X_raw.shape)

    # run emulation quantum-kmeans with metrics
    print("\n=== Running Quantum Emulation K-Means ===")
    labels_em, centroids, quantum_metrics = quantum_kmeans_emulation(X_raw, X_norm, k=args.k)
    
    # classical baseline with metrics
    print("\n=== Running Classical K-Means ===")
    labels_km, km_model, classical_metrics = classical_kmeans_with_metrics(X_raw, k=args.k)

    # Calculate clustering quality metrics
    print("\n=== Calculating Clustering Quality Metrics ===")
    quality_metrics = calculate_clustering_metrics(X_raw, labels_em, labels_km)

    # save cluster assignments to CSV
    out = df.copy()
    out['cluster_quantum_emulation'] = labels_em
    out['cluster_classical_kmeans'] = labels_km
    out_csv = os.path.splitext(os.path.basename(args.data))[0] + f"_clusters_k{args.k}.csv"
    out.to_csv(out_csv, index=False)
    print("Saved cluster assignments to", out_csv)

    # visualization
    out_png = os.path.splitext(os.path.basename(args.data))[0] + f"_cluster_compare_k{args.k}.png"
    plot_clusters_and_save(X_raw, labels_em, labels_km, out_png=out_png)

    # Print comprehensive analysis
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON ANALYSIS")
    print("="*60)
    
    print("\n--- Cluster Distribution ---")
    for i in range(args.k):
        print(f"Quantum cluster {i}: {np.sum(labels_em==i)} samples")
    for i in range(args.k):
        print(f"Classical cluster {i}: {np.sum(labels_km==i)} samples")
    
    print("\n--- Convergence Analysis ---")
    print(f"Quantum emulation converged in: {quantum_metrics['convergence_iterations']} iterations")
    print(f"Classical K-means converged in: {classical_metrics['convergence_iterations']} iterations")
    
    print("\n--- Runtime Comparison ---")
    print(f"Quantum emulation runtime: {quantum_metrics['runtime']:.4f} seconds")
    print(f"Classical K-means runtime: {classical_metrics['runtime']:.4f} seconds")
    print(f"Speedup ratio: {classical_metrics['runtime']/quantum_metrics['runtime']:.2f}x {'(quantum faster)' if quantum_metrics['runtime'] < classical_metrics['runtime'] else '(classical faster)'}")
    
    print("\n--- Clustering Quality Metrics ---")
    print(f"Quantum emulation inertia: {quantum_metrics['inertia']:.2f}")
    print(f"Classical K-means inertia: {classical_metrics['inertia']:.2f}")
    print(f"Quantum silhouette score: {quality_metrics['silhouette_quantum']:.4f}")
    print(f"Classical silhouette score: {quality_metrics['silhouette_classical']:.4f}")
    print(f"Agreement (Adjusted Rand Index): {quality_metrics['adjusted_rand_index']:.4f}")
    
    print("\n--- Performance Summary ---")
    better_silhouette = "Quantum" if quality_metrics['silhouette_quantum'] > quality_metrics['silhouette_classical'] else "Classical"
    better_inertia = "Quantum" if quantum_metrics['inertia'] < classical_metrics['inertia'] else "Classical"
    faster_convergence = "Quantum" if quantum_metrics['convergence_iterations'] < classical_metrics['convergence_iterations'] else "Classical"
    
    print(f"Better clustering quality (silhouette): {better_silhouette}")
    print(f"Better inertia (lower is better): {better_inertia}")
    print(f"Faster convergence: {faster_convergence}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Convergence Iterations', 'Runtime (s)', 'Inertia', 'Silhouette Score'],
        'Quantum_Emulation': [quantum_metrics['convergence_iterations'], 
                             quantum_metrics['runtime'], 
                             quantum_metrics['inertia'], 
                             quality_metrics['silhouette_quantum']],
        'Classical_KMeans': [classical_metrics['convergence_iterations'], 
                           classical_metrics['runtime'], 
                           classical_metrics['inertia'], 
                           quality_metrics['silhouette_classical']]
    })
    metrics_csv = os.path.splitext(os.path.basename(args.data))[0] + f"_metrics_k{args.k}.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved performance metrics to {metrics_csv}")

    # optional: compute one example swap-test similarity (very slow for many pairs)
    if args.use_qiskit:
        qiskit_mod, _, _, _ = try_import_qiskit()
        if qiskit_mod is None:
            print("Qiskit not installed or failed to import; skipping swap test.")
        else:
            print("\n=== Running Qiskit Swap Test ===")
            print("Running an example swap-test similarity (this is slow).")
            # pick first sample and centroid 0 (pad to power of two inside function)
            sample_vec = X_norm[0]
            centroid_vec = centroids[0]
            fid = swap_test_qiskit(sample_vec, centroid_vec, shots=2048)
            print(f"Swap-test estimated fidelity (|<psi|phi>|^2) between sample0 and centroid0: {fid:.4f}")
            print("Note: swap-test fidelity ≈ (inner-product)^2 for normalized amplitude vectors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='zip file containing a CSV (e.g. croprecommondation.zip)')
    parser.add_argument('--k', type=int, default=3, help='number of clusters')
    parser.add_argument('--use-qiskit', dest='use_qiskit', action='store_true', help='attempt to run swap-test with Qiskit (optional & slower)')
    args = parser.parse_args()
    main(args)
