# Graph Data Contract & CSR Specification — FraudGraph Intelligence (FGI)

**Module:** FGI-MOD-03 (Graph Representation)  
**Implementation Source:** [`backend/cpu_engine/csr_graph.py`](file:///D:/FraudGraph%20Intelligence/backend/cpu_engine/csr_graph.py)  
**Document Status:** Phase 3 Authoritative Specification  

---

## 1. Canonical Graph Data Structure (`CSRGraph`)

The canonical graph structure in FraudGraph Intelligence is the Compressed Sparse Row (CSR) representation. This format provides an optimal balance between low memory footprint ($O(N + M)$), constant-time degree retrieval, and direct memory alignment for CUDA warp-coalesced parallel neighbor reductions.

```
+----------------------------------------------------------------------------------------------------+
|                                    CSRGRAPH MEMORY CONTRACT                                        |
+----------------------------------------------------------------------------------------------------+
|  Field Name      | Data Type    | Shape                  | Semantics & Constraints                 |
+------------------+--------------+------------------------+-----------------------------------------+
|  row_ptr         | np.int32     | (num_nodes + 1,)       | Monotonically non-decreasing offsets.   |
|                  |              |                        | row_ptr[0] == 0, row_ptr[-1] == M.      |
|  col_idx         | np.int32     | (num_edges,)           | Target neighbor node indices.           |
|                  |              |                        | 0 <= col_idx[p] < num_nodes.            |
|  features        | np.float32   | (num_nodes, feat_dim)  | Dense feature vectors in row-major order.|
|                  |              |                        | All elements strictly finite.           |
|  num_nodes       | int          | Scalar                 | Total vertex count (len(row_ptr) - 1).  |
|  num_edges       | int          | Scalar                 | Total directed edges (len(col_idx)).    |
|  feature_dim     | int          | Scalar                 | Feature vector width (features.shape[1])|
|  is_directed     | bool         | Scalar                 | Explicit directionality flag.           |
|  node_id_to_idx  | dict         | {EntityID: NodeIndex}  | Bijective source mapping.               |
|  idx_to_node_id  | list         | [NodeIndex -> EntityID]| Reverse index lookup table.             |
|  metadata        | dict         | Key-Value JSON         | Provenance, policies, and configs.      |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Formal Invariants & Validation Rules

A graph is admitted into the FGI compute engine if and only if it satisfies all 8 programmatic invariants verified by `CSRGraph.validate()`:

1. **Base Offset Invariant:**
   $$\text{row\_ptr}[0] = 0$$
2. **Monotonicity Invariant:**
   $$\forall i \in [0, N-1], \quad \text{row\_ptr}[i+1] \ge \text{row\_ptr}[i]$$
3. **Edge Count Alignment:**
   $$\text{row\_ptr}[N] = M = \text{len}(\text{col\_idx})$$
4. **Index Range Validity:**
   $$\forall p \in [0, M-1], \quad 0 \le \text{col\_idx}[p] < N$$
5. **Feature Dimensional Consistency:**
   $$\text{features.shape} = (N, D)$$
6. **Floating-Point Finiteness:**
   $$\forall i \in [0, N-1], \; k \in [0, D-1], \quad \text{np.isfinite}(\text{features}[i, k]) == \text{True}$$
   *No $\text{NaN}$, $+\infty$, or $-\infty$ values are permitted.*
7. **Empty Neighborhood & Isolated Node Encoding:**
   If node $i$ has degree $d_i = 0$, its offset range is empty:
   $$\text{row\_ptr}[i+1] = \text{row\_ptr}[i]$$
8. **Undirected Symmetry Enforcement:**
   If $\text{is\_directed} == \text{False}$, then for every directed edge $(u, v) \in E$, there must exist a reciprocal directed edge $(v, u) \in E$.

---

## 3. Edge-Count Conventions

- **Directed Graphs ($\text{is\_directed} = \text{True}$):**
  $\text{num\_edges} = M = \text{len}(\text{col\_idx})$. Each entry $(u, v)$ represents a one-way relation from $u$ to $v$.
- **Undirected Graphs ($\text{is\_directed} = \text{False}$):**
  Edges are stored symmetrically in CSR. $\text{len}(\text{col\_idx})$ represents the total number of directed half-edges ($2 \times |E_{\text{undirected}}|$). The method `get_stats()["undirected_equivalent_edges"]` reports $\frac{M}{2}$.

---

## 4. Integer Widths & CUDA Memory Alignment

- **Offsets and Indices:** `np.int32` is selected as the canonical integer width for both `row_ptr` and `col_idx`.
  - *Capacity:* Supports up to $2^{31} - 1 = 2,147,483,647$ edges, far exceeding single-GPU VRAM capacity.
  - *Hardware Alignment:* Maps directly to native 32-bit CUDA `int` (`int4` vector loads), avoiding the 64-bit address register pressure that reduces warp occupancy.
- **Node Features:** `np.float32` (IEEE 754 single precision) in row-major contiguous memory.
  - *Hardware Alignment:* Compatible with 32-bit single-precision floating-point ALUs and allows 32 consecutive warp threads to issue aligned 128-byte DRAM transactions.

---

## 5. Memory Footprint Estimation

The exact physical memory footprint of a `CSRGraph` instance is computed as:
$$\text{Memory}_{\text{bytes}} = (N + 1) \times 4 + M \times 4 + N \times D \times 4 \quad \text{bytes}$$

$$\text{Memory}_{\text{MB}} = \frac{\text{Memory}_{\text{bytes}}}{1024 \times 1024}$$

### Benchmark Tier Footprints:
- **Toy Graph ($N=5, M=8, D=4$):** 136 bytes ($\approx 0.0001 \text{ MB}$).
- **Tier A ($N=10,000, M=94,945, D=32$):** $1,699,784 \text{ bytes} \approx 1.62 \text{ MB}$.
- **Tier B ($N=50,000, M=350,000, D=64$):** $14,400,004 \text{ bytes} \approx 13.73 \text{ MB}$.
- **Tier C ($N=200,000, M=1,800,000, D=128$):** $110,400,004 \text{ bytes} \approx 105.29 \text{ MB}$.
*All tiers fit comfortably within the 1907 MiB free VRAM of the local NVIDIA GeForce MX450.*
