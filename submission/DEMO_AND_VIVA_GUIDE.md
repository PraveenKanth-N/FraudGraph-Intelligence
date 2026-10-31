# FraudGraph Intelligence (FGI): Demonstration & Viva Defense Guide

**Document Status:** Authoritative Capstone Defense & Demonstration Guide  
**Course:** CSI4007 — GPU Programming Capstone  
**Institution:** Vellore Institute of Technology (VIT), Vellore Campus  
**Workspace:** `D:\FraudGraph Intelligence`  
**Git Baseline Commit:** `8afa28cf1172c006f0242f530267e8cc4466f9d6` (Branch `main`)  
**Audience:** Capstone Evaluation Panel, Faculty Evaluators, Technical Reviewers  

---

## PART 1: Structured 8-Minute Demonstration Script

This walkthrough provides a clear, professional presentation outline designed for an 8-minute demonstration slot.

| Elapsed Time | Demonstration Phase | Screen / UI Target | Live Action & Key Talking Points |
| :--- | :--- | :--- | :--- |
| **0:00 – 1:00** | **Motivation & Problem Statement** | Title Slide / Overview | - Introduce project title: *Memory-Aware CUDA Acceleration of Graph-Based Financial Fraud Detection Using Parallel Neighbor Aggregation*.<br>- Explain why financial institutions struggle with relational fraud (money mule rings, synthetic identity theft, shared devices/cards).<br>- Contrast tabular detectors (blind to relational linkages) with graph neighbor aggregation.<br>- Define the core engineering challenge: power-law graph skew causes severe SIMT warp divergence and PCIe transfer bottlenecks on GPUs. |
| **1:00 – 2:00** | **System Architecture & Telemetry** | `/` (Overview Page) | - Point to live KPI telemetry cards: Node count (10,000), Directed edges (231,984), Fraud prevalence (3.50%).<br>- Open the **System Health Drawer** (top-right pill): highlight live hardware detection (`local_gpu_detected: None` on host, `cuda_compiler_status: NO_GPU_DETECTED_COLAB_READY`).<br>- Explain the dual-environment architecture: CPU/FastAPI runs locally; low-level CUDA kernels were executed and profiled on an NVIDIA Tesla T4 in Google Colab. |
| **2:00 – 3:00** | **Dataset Registry & Integrity** | `/datasets` | - Inspect `synthetic_fraud_10k`: highlight the cryptographic SHA-256 verification pill (`aecafe3b...`).<br>- Show dataset characteristics: 10k nodes, 2.24 MB CSR footprint, 1,155 isolated nodes (11.5%), max in-degree of 252.<br>- Point out the strict chronological 70/15/15 temporal split (zero future-to-past edges). |
| **3:00 – 4:15** | **Topological Subgraph Explorer** | `/graph` | - Demonstrate the live interactive Breadth-First Search (BFS) neighborhood explorer.<br>- Select Center Node `0` with Depth `2`: observe how the backend extracts the local neighborhood from CSR arrays.<br>- Show fraudster clustering: point to red fraud nodes connected to shared credit cards/devices.<br>- Show the defensive truncation pill when edge limits are reached (`is_truncated: true`), preventing browser DOM overload. |
| **4:15 – 5:15** | **Transaction Ledger & Filtering** | `/transactions` | - Demonstrate server-side pagination over the 10,000-transaction table.<br>- Toggle the `Fraud Only` filter: observe immediate filtering down to the 350 ground truth fraud cases.<br>- Highlight transactional feature columns (amount, velocity, timestamp) and show how graph labels link directly to graph entities. |
| **5:15 – 6:30** | **CUDA Acceleration Lab** | `/cuda-lab` | - **The Core HPC Defense:** Walk the panel through the 4 CUDA kernel strategies profiled on the Tesla T4:<br>1. *Baseline (Thread/Node):* Suffers catastrophic SIMT divergence ($6.98$ ms on skewed graph).<br>2. *Warp-Coalesced (1 Warp/Node):* Satisfies memory in aligned 128-byte transactions, yielding a **$31.8\times$ kernel speedup** ($0.22$ ms).<br>3. *PCIe Transfer Breakdown:* Show that PCIe transfers ($0.91$ ms) dominate kernel compute ($0.068$ ms) by $13.5\times$.<br>4. *Pinned Streams:* Asynchronous double-buffering cuts end-to-end latency by **$3.5\times$** ($0.99 \to 0.28$ ms).<br>5. *Degree-Aware Skew:* Explain the nuanced trade-off: **$3.4\times$ gain** on MEAN, but a **$2.19\times$ regression** on SUM. Explain that Warp 0 was assigned serial accumulation to guarantee float32 non-associative numerical equivalence ($\text{atol} \le 10^{-5}$), leaving 7 of 8 warps idle in hub blocks! |
| **6:30 – 7:30** | **Downstream Fraud Evaluation** | `/evaluation` | - Explain why PR-AUC is the gold standard for 3.5% fraud imbalance (ROC-AUC is artificially inflated to 0.78 by true negatives).<br>- Highlight the ML operational trade-offs:<br>1. **Logistic Regression:** MAX enrichment boosts test PR-AUC from 0.3088 to **0.3315** and recall from 37.7% to **45.3%**, though precision declines from 31.7% to 25.5% and F1 from 0.345 to 0.327 (a high-sensitivity alert filter).<br>2. **Random Forest:** MEAN enrichment cuts false alerts from 138 down to 30 (**78.3% false-positive reduction**) and doubles precision from 16.9% to **33.3%**, but recall falls from 52.8% to **28.3%** (FN increases from 25 to 38).<br>3. Show the catastrophic failure of SUM (PR-AUC drops $>50\%$), proving that raw sums conflate high degree with fraud. |
| **7:30 – 8:00** | **Summary, Defense & Limitations** | Reports / Summary | - Conclude with scientific transparency:<br>- Confirmed 188 backend tests passed; 24 frontend Node tests passed; clean Vite build.<br>- Transparently disclose that local GPU is absent and WebGPU browser shaders are deferred.<br>- Open the floor for viva examination. |

---

## PART 2: Comprehensive Viva Defense (12 Key Questions & Evidence-Based Answers)

### Q1: Why is graph representation necessary for financial fraud detection instead of standard tabular XGBoost?
**Answer:**  
Standard tabular classifiers evaluate each transaction in isolation or aggregate historical statistics across a single account. However, modern financial fraud—such as organized money mule syndicates, bust-out fraud, and credit card theft—operates across **relational networks** where collusive actors share synthetic devices, IP subnets, bank accounts, and physical addresses.  
When an attacker engages in *camouflage* (making low-amount, benign-looking transactions to evade tabular rules), their individual tabular feature vector appears legitimate. By projecting transactions into a bipartite relational graph sharing card and device identifiers, neighbor aggregation propagates topological signals from known fraudulent entities directly into the target node's feature space. As our empirical evaluation shows, enriching raw features with neighbor context boosted Logistic Regression PR-AUC from 0.3088 to **0.3315** and reduced Random Forest false positives by **78.3%**.

---

### Q2: Why did you choose Compressed Sparse Row (CSR) format instead of an Adjacency Matrix or Edge List?
**Answer:**  
The choice of CSR is dictated by memory complexity, cache locality, and GPU memory bus alignment:
1. **Memory Efficiency:** An adjacency matrix for $N=10,000$ transactions would require $N^2 \times 4$ bytes = $400$ MB, where $>99.7\%$ of entries are zero (density is only $0.00232$). In contrast, CSR stores only non-zero edges using two 1D integer arrays: `row_ptr` ($N+1 = 10,001$ elements) and `col_idx` ($M = 231,984$ elements), consuming just **$2.24$ MB** of memory (a $>178\times$ memory reduction).
2. **Contiguous Neighbor Slicing:** Unlike an unsorted Edge List (COO) where finding a node's neighbors requires scanning the entire edge array or executing an irregular binary search, CSR guarantees that all inbound neighbors of node $i$ reside in contiguous memory between indices `row_ptr[i]` and `row_ptr[i+1]`.
3. **GPU Bus Coalescing:** Contiguous neighbor indices allow CUDA threads to execute coalesced 128-byte global memory transactions, saturating GPU memory bandwidth.

---

### Q3: What mathematical operators are computed during neighbor aggregation, and what do they represent semantically?
**Answer:**  
We evaluate three core aggregation operators across node feature vectors $X \in \mathbb{R}^{N \times D}$:
1. **MEAN Aggregation ($H_{i} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} X_j$):** Computes the centroid or baseline behavioral profile of connected neighbors. Normalization prevents high-degree hub nodes from distorting the representation. Semantically, it captures whether an account routinely transacts with high-risk counterparts.
2. **MAX Aggregation ($H_{i} = \max_{j \in \mathcal{N}(i)} X_j$):** Extracts the extreme or worst-case attribute across all connected entities (e.g. maximum transaction velocity, maximum amount anomaly, highest foreign risk score). Semantically, a single fraudulent connection in a money mule ring can expose the entire cluster.
3. **SUM Aggregation ($H_{i} = \sum_{j \in \mathcal{N}(i)} X_j$):** Unnormalized volume accumulator. In our experiments, SUM performed poorly because it acts as an uncalibrated proxy for node in-degree, leading to false positives on high-volume benign merchants.

---

### Q4: Why did the naive thread-per-node CUDA kernel perform poorly on the power-law skewed graph?
**Answer:**  
The naive kernel assigns 1 CUDA thread to 1 node $i$, which iterates over $j \in [row\_ptr[i], row\_ptr[i+1])$ and accumulates feature dimension $d$. On a power-law graph, this design causes two severe micro-architectural bottlenecks:
1. **SIMT Branch Divergence:** CUDA executes threads in lockstep groups of 32 called *warps*. If one thread in a warp is assigned a hub node with high degree (such as $d_i = 252$ in the fraud graph, or up to $d_i = 6,340$ in the skewed stress fixture `synthetic_skewed_10k`) while the remaining 31 threads have $d_i = 6$ neighbors, the 31 threads finish their loop in 6 iterations but are forced to stall and remain idle while the single thread executes the remaining iterations. Warp efficiency drops to $<3\%$.
2. **Uncoalesced Memory Access:** When adjacent threads in a warp access $X[j, d]$, they access completely different neighbor indices $j_0, j_1, \dots, j_{31}$ scattered non-contiguously throughout DRAM. The memory controller must issue separate 32-byte memory transactions for each thread, wasting $>75\%$ of memory bus bandwidth.  
As a result, latency on the skewed graph degraded to **$6.98$ ms**.

---

### Q5: How does the Warp-Coalesced kernel solve this memory bottleneck?
**Answer:**  
In our `fgi_coalesced_kernels.cu` implementation, we map **1 entire 32-thread warp to 1 node $i$**:
- Thread lane `k` (where $k = threadIdx.x \pmod{32}$) is responsible for feature dimension $d = k$.
- When the warp iterates over neighbor $j$, all 32 threads simultaneously read $X[j, 0], X[j, 1], \dots, X[j, 31]$.
- Because feature vectors are laid out contiguously in row-major order, these 32 4-byte floats form an aligned 128-byte block in DRAM.
- The GPU memory subsystem satisfies the entire warp's read in **a single 128-byte transaction**, achieving peak memory bus efficiency.  
This lifted effective memory bandwidth from 43.4 GB/s to **175.3 GB/s** and reduced kernel execution latency on the skewed graph from $6.98$ ms to **$0.22$ ms** (a **$31.8\times$ speedup**).

---

### Q6: Why did Degree-Aware dispatch accelerate MEAN and MAX, but REGRESS SUM on the skewed graph?
**Answer:**  
This is one of the most critical engineering findings in our research:
- In `fgi_degree_aware_kernels.cu`, nodes are partitioned into buckets: standard nodes ($d_i \le 32$) and hub nodes ($d_i > 32$). High-degree hub nodes are assigned to an entire cooperative thread block (256 threads = 8 warps) using block-level reduction (`hub_degree_block_kernel`).
- For **MEAN and MAX**, all 8 warps cooperatively interleave edge reads across the super-hub in parallel (`p += 8`), stage partial results into shared memory (`s_warp_acc[8][32]`), and perform a clean 8-way reduction on Warp 0. This accelerated MEAN by **$3.4\times$** ($0.065$ ms) and MAX by **$2.1\times$** ($0.111$ ms) over the monolithic coalesced kernel.
- For **SUM**, however, cooperative reordering across 6,340 edges accumulates single-precision floating-point rounding drift due to the non-associativity of IEEE 754 float32 addition: $(a + b) + c \neq a + (b + c)$. In initial Colab runs, this drift ($3.97 \times 10^{-4}$) failed the project's strict numerical verification gate ($\text{atol} \le 10^{-5}$). To ensure strict mathematical parity with the CPU reference, the implementation enforces single-warp serial accumulation on Warp 0 (`if (warp_id == 0)`).
- **Architectural Impact:** Enforcing `if (warp_id == 0)` causes 7 out of 8 warps (224 threads out of 256) to sit completely idle while Warp 0 loops through all 6,340 edges. This severely under-occupies the SM and prevents latency hiding, providing a clear code-observed explanation for the measured **$2.19\times$ kernel slowdown** ($0.212 \to 0.464$ ms). Zero atomic operations were present in this kernel path.

---

### Q7: What is the PCIe data transfer bottleneck, and how did Pinned Memory Streams resolve it?
**Answer:**  
On our Tier A baseline, measuring isolated kernel compute with `cudaEvent_t` revealed a kernel latency of just **$0.068$ ms**. However, transferring the CSR structures and features across the PCIe 3.0 x16 bus required **$0.58$ ms** Host-to-Device (H2D) and **$0.33$ ms** Device-to-Host (D2H), totaling **$0.91$ ms**.
**PCIe transfer latency was $13.5\times$ larger than the actual GPU kernel computation!**  
To solve this, we implemented asynchronous multi-stream double-buffering in `main_streams.cu`:
1. Host memory is allocated as page-locked (pinned) memory via `cudaHostAlloc`, allowing the GPU Direct Memory Access (DMA) engine to stream data without host CPU staging.
2. The graph is split into 4 chunks across 4 concurrent `cudaStream_t` queues. While Stream 2 is transferring chunk 2, Stream 1 is executing the kernel on chunk 1, and Stream 0 is transferring back chunk 0.  
This overlapped PCIe bus transfers with GPU kernel compute, cutting end-to-end latency from $0.99$ ms down to **$0.28$ ms** (a **$3.5\times$ end-to-end speedup**).

---

### Q8: Why is PR-AUC the primary evaluation metric instead of Accuracy or ROC-AUC?
**Answer:**  
Financial fraud detection suffers from extreme class imbalance (in our dataset, exactly **$3.50\%$** prevalence; in real banking, often $<0.1\%$).
1. **Accuracy is Misleading:** A naive dummy model predicting "0 (Licit)" for every transaction achieves **$96.5\%$ accuracy** while detecting exactly zero fraud.
2. **ROC-AUC is Overly Optimistic:** The Receiver Operating Characteristic plots True Positive Rate against False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$). Because the number of true negatives ($\text{TN} = 1,447$) dwarfs the positives, even a large surge of false positives (e.g. 100 false alerts) results in an FPR of only $0.06$, yielding a deceptively high ROC-AUC of $\sim 0.78$.
3. **PR-AUC Evaluates Operational Precision:** Precision-Recall curves plot Precision ($\frac{\text{TP}}{\text{TP} + \text{FP}}$) against Recall ($\frac{\text{TP}}{\text{TP} + \text{FN}}$). Precision directly penalizes false alerts that waste human fraud investigator time. In our test set, a random baseline yields a PR-AUC of only $0.0353$, making our achieved PR-AUC of **$0.3315$** an $>8\times$ improvement over chance.

---

### Q9: How was temporal data leakage prevented in your train/validation/test split?
**Answer:**  
Temporal leakage occurs when future transactions or graph relationships leak into the historical training phase, creating artificially inflated evaluation metrics that collapse in production. We prevented leakage via two strict guarantees:
1. **Chronological Splitting:** Transactions were ordered strictly by physical timestamp ($t_0 \dots t_N$) and split chronologically: the first 70% ($t \le 21.0$ days) for Training, the next 15% ($21.0 < t \le 25.5$ days) for Validation, and the final 15% ($25.5 < t \le 30.0$ days) for Test.
2. **Causal In-Degree Lookback:** Graph edges represent relational identity links over a sliding historical lookback window of 7.0 days ($604,800$ s). Edges are strictly directed forward in time ($t_{\text{source}} < t_{\text{target}}$). No future transaction can ever connect backward to an earlier transaction, ensuring zero future-to-past information leakage.

---

### Q10: What are the weaknesses of synthetic fraud datasets compared to real-world financial networks?
**Answer:**  
While `synthetic_fraud_10k` provides a controlled, reproducible testbed with verified ground truth, several key differences exist relative to production banking networks:
1. **Scale:** Real-world banking platforms (e.g. Visa, Alipay) process hundreds of millions of daily transactions, requiring distributed multi-GPU partitioning (e.g. via NCCL/NVLink).
2. **Adversarial Camouflage Sophistication:** In real systems, professional fraud syndicates dynamically alter transaction timing, split amounts below reporting thresholds (smurfing), and cycle through burner devices.
3. **Graph Homophily Violations:** Real financial networks are multi-relational heterographs with heterogeneous edge types (cards, devices, IP subnets, bank accounts, merchant categories), whereas our synthetic graph projects multi-relations into a homogeneous CSR adjacency matrix.

---

### Q11: Why is kernel-only speedup fundamentally different from end-to-end system speedup?
**Answer:**  
Kernel-only speedup measures the time spent strictly inside GPU execution units on the Streaming Multiprocessors, isolated between `cudaEventRecord` start and stop points after data is already resident in GPU VRAM.  
In contrast, end-to-end system speedup encompasses:
1. Host memory allocation and graph serialization into CSR arrays.
2. PCIe Host-to-Device memory copy (`cudaMemcpy` H2D).
3. CUDA kernel launch and execution.
4. PCIe Device-to-Host memory copy (`cudaMemcpy` D2H).
5. Downstream host ingestion and model scoring.  
Reporting only kernel speedup ($>30\times$) without PCIe transfer overhead creates a misleading perception of real-world latency. As Amdahl's Law dictates, when kernel compute is accelerated to $0.017$ ms, the unaccelerated PCIe transfer ($0.91$ ms) becomes the dominant bottleneck ($>98\%$ of execution time).

---

### Q12: What components are implemented versus deferred in FraudGraph Intelligence?
**Answer:**  
- **Implemented & Verified:**
  - 4 CUDA kernel architectures (Baseline, Warp-Coalesced, Degree-Aware, Pinned Streams) compiled and profiled on Tesla T4.
  - Complete CPU reference and SciPy vectorized SpMM engines with 188 passing backend pytest tests.
  - Complete FastAPI backend (6 endpoints) with BFS subgraph extraction and transaction pagination.
  - Full React/Vite analytical dashboard (8 pages) verified via 24 Node tests, TypeScript compilation, and production build.
  - Downstream ML classification pipelines across 5 representations with validation-frozen thresholds.
- **Explicitly Deferred (Future Work):**
  - Live in-browser WebGPU shader execution (the UI renders verified Colab GPU benchmarks and live CPU/API endpoints).
  - Multi-GPU distributed partitioning (NCCL).
  - Continuous streaming graph ingestion with real-time dynamic CSR pointer updates.
  - Headless browser E2E test automation (Playwright/Cypress).

---
*End of Demo & Viva Defense Guide.*
