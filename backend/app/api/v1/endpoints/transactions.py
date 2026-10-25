import os
import hashlib
import tempfile
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import numpy as np

router = APIRouter()

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
CANONICAL_FIXTURE_PATH = os.path.join(WORKSPACE_ROOT, "data", "fixtures", "synthetic_fraud_10k.npz")


def _is_safe_path(path: str) -> bool:
    """Ensure path is within WORKSPACE_ROOT or an authorized system temp directory."""
    abs_path = os.path.abspath(path)
    temp_dir = os.path.abspath(tempfile.gettempdir())

    in_workspace = False
    try:
        in_workspace = os.path.commonpath([abs_path, WORKSPACE_ROOT]) == WORKSPACE_ROOT
    except ValueError:
        pass

    in_temp = False
    try:
        in_temp = os.path.commonpath([abs_path, temp_dir]) == temp_dir
    except ValueError:
        pass

    return in_workspace or in_temp


class TransactionItem(BaseModel):
    tx_id: int
    timestamp_sec: float
    fraud_label: int
    is_fraud: bool
    risk_tier: str
    in_degree: int
    feature_dim: int
    feature_sample: List[float]
    source_card: Optional[str] = None
    target_device: Optional[str] = None
    notes: str = "Relational entity strings were discarded during CSR compression; graph topology preserves entity relations via directed edges."


class TransactionListResponse(BaseModel):
    status: str
    total_dataset_records: int
    total_fraud_records: int
    total_licit_records: int
    filtered_count: int
    offset: int
    limit: int
    returned_count: int
    has_next: bool
    has_prev: bool
    current_filter: str
    ordering: str = "original_transaction_index_chronological"
    items: List[TransactionItem]
    schema_notes: str = "source_card and target_device are omitted in CSR graph representation; in_degree indicates connected transactions in historical graph."
    provenance_sha256: str


def _load_transactions_arrays(path: str):
    """Safely loads and validates NPZ arrays for transaction pagination."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Dataset fixture file not found: {os.path.basename(path)}")

    try:
        with open(path, "rb") as fp:
            file_bytes = fp.read()
            sha256 = hashlib.sha256(file_bytes).hexdigest()

        with np.load(path) as data:
            required_keys = ["row_ptr", "timestamps", "labels", "features"]
            for k in required_keys:
                if k not in data:
                    raise ValueError(f"Missing required key '{k}' in NPZ file")

            row_ptr = np.array(data["row_ptr"])
            timestamps = np.array(data["timestamps"])
            labels = np.array(data["labels"])
            features = np.array(data["features"])

        if len(labels) < 1:
            raise ValueError("labels array is empty")

        return row_ptr, timestamps, labels, features, sha256
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Dataset fixture file is corrupt or invalid: {str(e)}")


@router.get("", response_model=TransactionListResponse)
@router.get("/", response_model=TransactionListResponse, include_in_schema=False)
def get_transactions(
    offset: int = Query(0, ge=0, description="Record pagination offset (>= 0)"),
    limit: int = Query(50, ge=1, le=100, description="Items per page (1 to 100)"),
    filter: str = Query("all", description="Label filter: 'all', 'fraud', or 'licit'"),
    npz_path: Optional[str] = Query(None, description="Optional path to NPZ dataset fixture"),
):
    """Get paginated, filtered transaction records derived authentically from dataset fixture."""
    filter_lower = filter.strip().lower()
    if filter_lower not in ("all", "fraud", "licit"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filter '{filter}'. Must be one of 'all', 'fraud', or 'licit'."
        )

    target_path = npz_path or CANONICAL_FIXTURE_PATH
    if not _is_safe_path(target_path):
        raise HTTPException(status_code=400, detail="npz_path escapes authorized workspace boundaries")

    row_ptr, timestamps, labels, features, sha256 = _load_transactions_arrays(target_path)
    total_records = len(labels)

    # Derive totals dynamically from actual labels array
    fraud_indices = np.where(labels == 1)[0]
    licit_indices = np.where(labels == 0)[0]
    total_fraud = int(len(fraud_indices))
    total_licit = int(len(licit_indices))

    # Apply filter before pagination
    if filter_lower == "all":
        selected_indices = np.arange(total_records)
    elif filter_lower == "fraud":
        selected_indices = fraud_indices
    else:  # "licit"
        selected_indices = licit_indices

    filtered_count = len(selected_indices)

    # Slice page
    if offset >= filtered_count:
        page_indices = []
        has_next = False
    else:
        page_indices = selected_indices[offset : offset + limit]
        has_next = (offset + limit) < filtered_count

    has_prev = offset > 0

    feat_dim = int(features.shape[1]) if features.ndim > 1 else 1

    items: List[TransactionItem] = []
    for idx in page_indices:
        i = int(idx)
        lbl = int(labels[i])
        is_frd = (lbl == 1)
        deg = int(row_ptr[i + 1] - row_ptr[i])
        feat_sample = [round(float(v), 4) for v in features[i, :4]]

        items.append(
            TransactionItem(
                tx_id=i,
                timestamp_sec=round(float(timestamps[i]), 2),
                fraud_label=lbl,
                is_fraud=is_frd,
                risk_tier="CONFIRMED_FRAUD" if is_frd else "LICIT_CLEARED",
                in_degree=deg,
                feature_dim=feat_dim,
                feature_sample=feat_sample,
                source_card=None,
                target_device=None,
            )
        )

    return TransactionListResponse(
        status="success",
        total_dataset_records=total_records,
        total_fraud_records=total_fraud,
        total_licit_records=total_licit,
        filtered_count=filtered_count,
        offset=offset,
        limit=limit,
        returned_count=len(items),
        has_next=has_next,
        has_prev=has_prev,
        current_filter=filter_lower,
        ordering="original_transaction_index_chronological",
        items=items,
        schema_notes="source_card and target_device are omitted in CSR graph representation; in_degree indicates connected transactions in historical graph.",
        provenance_sha256=sha256,
    )
