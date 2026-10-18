"""
Temporal Transaction Graph Builder for Financial Tabular Datasets
Module: FGI-MOD-02 & FGI-MOD-03
Course: VIT Vellore — CSI4007 GPU Programming

Enforces strict temporal ordering, prevents future-neighbor leakage, rejects null sentinels,
and quarantines ground-truth fraud labels (isFraud) from graph feature and topology construction.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
from backend.cpu_engine.csr_graph import CSRGraph


class TemporalTransactionGraphBuilder:
    """
    Constructs a verified Compressed Sparse Row (CSR) transaction graph from tabular records.
    
    Leakage Prevention Guarantees:
    1. Temporal Causality: An edge is directed from u to v only if TransactionDT(u) <= TransactionDT(v).
       Transactions can never aggregate information from future transactions.
    2. Label Quarantine: 'isFraud' is strictly isolated as an evaluation target and NEVER used in
       feature generation or edge matching.
    3. Null Sentinel Rejection: Missing/null keys (None, NaN, empty strings) are strictly excluded
       so that absent identifiers do not create artificial dense cliques.
    """

    def __init__(
        self,
        relational_keys: Sequence[str] = ("card1", "P_emaildomain"),
        temporal_window_seconds: float = 7 * 86400.0,  # 7-day sliding window
        max_transactions: int | None = None,
        feature_columns: Sequence[str] | None = None,
    ) -> None:
        self.relational_keys = list(relational_keys)
        self.temporal_window_seconds = float(temporal_window_seconds)
        self.max_transactions = max_transactions
        self.feature_columns = list(feature_columns) if feature_columns else None

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        is_directed: bool = True,
    ) -> tuple[CSRGraph, np.ndarray | None]:
        """
        Build a CSRGraph from an in-memory Pandas DataFrame.
        
        Args:
            df: DataFrame containing transaction records.
            is_directed: Must be True for temporal causality (earlier -> later).
            
        Returns:
            Tuple of (CSRGraph, labels_array or None).
        """
        if "TransactionID" not in df.columns:
            raise ValueError("Input DataFrame missing required primary key: 'TransactionID'")
        if "TransactionDT" not in df.columns:
            raise ValueError("Input DataFrame missing required timestamp column: 'TransactionDT'")

        # Make copy and limit rows if specified
        work_df = df.copy()
        if self.max_transactions is not None and len(work_df) > self.max_transactions:
            work_df = work_df.iloc[:self.max_transactions]

        # 1. Sort strictly by timestamp for temporal causality
        work_df.sort_values(by=["TransactionDT", "TransactionID"], inplace=True)
        work_df.reset_index(drop=True, inplace=True)

        num_nodes = len(work_df)
        transaction_ids = work_df["TransactionID"].tolist()
        timestamps = work_df["TransactionDT"].to_numpy()

        # Extract ground truth labels if present (quarantined)
        labels: np.ndarray | None = None
        if "isFraud" in work_df.columns:
            labels = work_df["isFraud"].to_numpy(dtype=np.int32)

        # 2. Extract and sanitize feature matrix
        excluded_cols = {"TransactionID", "isFraud", "TransactionDT"}
        if self.feature_columns:
            candidate_cols = [c for c in self.feature_columns if c in work_df.columns and c not in excluded_cols]
        else:
            # Auto-select numeric columns excluding keys and labels
            candidate_cols = [
                c for c in work_df.columns
                if c not in excluded_cols and pd.api.types.is_numeric_dtype(work_df[c])
            ]

        if len(candidate_cols) > 0:
            feat_df = work_df[candidate_cols].copy()
            # Median imputation for missing features
            for col in candidate_cols:
                median_val = feat_df[col].median()
                fill_val = float(median_val) if pd.notna(median_val) else 0.0
                feat_df[col] = feat_df[col].fillna(fill_val)
            raw_features = feat_df.to_numpy(dtype=np.float32, copy=True)
            features = np.nan_to_num(raw_features, copy=True, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        else:
            # Fallback 1D normalized amount or zero feature
            if "TransactionAmt" in work_df.columns:
                amt = work_df["TransactionAmt"].fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1)
                features = amt
            else:
                features = np.zeros((num_nodes, 16), dtype=np.float32)

        # 3. Construct edges based on relational keys
        edge_list: list[tuple[int, int]] = []
        for key in self.relational_keys:
            if key not in work_df.columns:
                continue

            # Group indices by key value, strictly discarding null sentinels
            valid_mask = work_df[key].notna() & (work_df[key] != "")
            valid_indices = np.where(valid_mask)[0]

            key_to_indices: dict[Any, list[int]] = {}
            for idx in valid_indices:
                val = work_df.iat[idx, work_df.columns.get_loc(key)]
                key_to_indices.setdefault(val, []).append(idx)

            # Build temporal edges within each key group
            for val, indices in key_to_indices.items():
                if len(indices) <= 1:
                    continue
                # Indices are already chronologically ordered
                for i in range(len(indices)):
                    u = indices[i]
                    t_u = timestamps[u]
                    for j in range(i + 1, len(indices)):
                        v = indices[j]
                        t_v = timestamps[v]
                        # Check sliding temporal window
                        if (t_v - t_u) <= self.temporal_window_seconds:
                            edge_list.append((u, v))
                            if not is_directed:
                                edge_list.append((v, u))
                        else:
                            # Because indices are sorted by timestamp, subsequent transactions will also exceed window
                            break

        metadata: dict[str, Any] = {
            "builder": "TemporalTransactionGraphBuilder",
            "relational_keys": self.relational_keys,
            "temporal_window_seconds": self.temporal_window_seconds,
            "feature_columns_used": candidate_cols,
            "has_labels": labels is not None,
            "is_directed": is_directed,
        }

        graph = CSRGraph.from_edge_list(
            num_nodes=num_nodes,
            edge_list=edge_list,
            features=features,
            feature_dim=features.shape[1],
            is_directed=is_directed,
            allow_self_loops=False,
            deduplicate=True,
            sort_neighbors=True,
            idx_to_node_id=transaction_ids,
            node_id_to_idx={tid: i for i, tid in enumerate(transaction_ids)},
            metadata=metadata,
        )

        return graph, labels

    @staticmethod
    def check_raw_dataset_availability(raw_dir: str | Path = "data/raw") -> dict[str, Any]:
        """
        Check if raw external dataset files (e.g. IEEE-CIS) are present locally.
        Does NOT download files automatically.
        """
        path = Path(raw_dir)
        files = list(path.glob("*")) if path.exists() else []
        file_names = [f.name for f in files if f.name != ".gitkeep"]

        has_ieee_train = any("train_transaction" in f for f in file_names)
        has_elliptic = any("elliptic" in f.lower() for f in file_names)

        return {
            "raw_dir_exists": path.exists(),
            "detected_files": file_names,
            "ieee_cis_available": has_ieee_train,
            "elliptic_available": has_elliptic,
            "instructions": (
                "To ingest real IEEE-CIS data, place 'train_transaction.csv' and 'train_identity.csv' in 'data/raw/'. "
                "The pipeline operates deterministically with synthetic fixtures if raw files are absent."
            )
        }
