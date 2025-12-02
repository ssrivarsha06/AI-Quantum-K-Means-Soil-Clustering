# Quantum KMeans Prototype

Hybrid **Quantum-Classical KMeans Clustering** prototype in Python.
This project emulates a quantum-style KMeans algorithm using normalized amplitude vectors and optionally integrates a Qiskit-based **swap-test** for small-scale quantum similarity computation.

---

## Features

* Quantum KMeans **emulation** (amplitude-style normalization, inner-product similarity).
* Classical KMeans baseline for comparison.
* Automatic **feature preprocessing**: encoding categorical variables, scaling, normalization.
* Computes clustering **quality metrics**:

  * Inertia
  * Silhouette Score
  * Adjusted Rand Index (quantum vs classical agreement)
* Generates **visualizations** (PCA 2D scatter plots).
* Optional **Qiskit swap-test** for fidelity estimation between sample and centroid.
* Saves clustering results and metrics as CSV files.

---

## Dependencies

* Python 3.8+
* Required Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

* Optional (for Qiskit swap-test):

```bash
pip install qiskit qiskit-aer
```

---

## Usage

1. Place your dataset ZIP (CSV inside) in the project folder.
2. Run the script:

```bash
python quantum_kmeans_prototype.py --data <dataset>.zip --k <num_clusters>
```

Example:

```bash
python quantum_kmeans_prototype.py --data croprecommondation.zip --k 3
```

* To enable **Qiskit swap-test**:

```bash
python quantum_kmeans_prototype.py --data croprecommondation.zip --k 3 --use-qiskit
```

---

## Outputs

* Cluster assignments CSV:

```
<dataset>_clusters_k<k>.csv
```

* Performance metrics CSV:

```
<dataset>_metrics_k<k>.csv
```

* Cluster comparison visualization (PCA 2D plot):

```
<dataset>_cluster_compare_k<k>.png
```

---

## Project Structure

```
QuantumKMeansProject/
├─ quantum_kmeans_prototype.py    # main script
├─ croprecommondation.zip         # sample dataset (CSV inside)
├─ nutrients.zip                  # optional dataset
├─ .gitignore
└─ README.md
```

---

## Notes

* The **venv/** folder and other temporary files are ignored in Git.
* Swap-test is **slow** for large datasets; recommended only for small sample-centroid comparisons.
* The quantum emulation uses **normalized vectors** to simulate amplitude-based similarity.

---

## References

* [Scikit-learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
* [Qiskit Swap Test Tutorial](https://qiskit.org/documentation/)

---

