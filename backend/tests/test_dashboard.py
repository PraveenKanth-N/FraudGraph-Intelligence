from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert data["service"] == "FraudGraph Intelligence"

def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "compute_engine" in data
    assert "local_gpu_detected" in data["compute_engine"]

def test_dashboard_summary():
    response = client.get("/api/v1/dashboard/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert data["project"] == "FraudGraph Intelligence"

    # Verify genuine dataset fields
    ds = data["dataset"]
    assert ds["name"] == "synthetic_fraud_10k"
    assert ds["num_nodes"] == 10000
    assert ds["num_edges"] == 231984
    assert ds["feature_dim"] == 32
    assert ds["fraud_count"] == 350
    assert ds["fraud_rate_pct"] == 3.5
    assert ds["is_synthetic"] is True

    # Verify evaluation summary
    ev = data["evaluation"]
    assert ev["top_model"] == "Logistic Regression (max)"
    assert ev["top_test_pr_auc"] > 0
    assert ev["top_test_roc_auc"] > 0
    assert "ADR-007" in ev["adr_007_notice"]

    # Verify benchmark summary
    bm = data["benchmarks"]
    assert bm["session_id"] == "session_20261024_230008_smoke"
    assert bm["total_runs"] == 36
    assert bm["measured_cpu_runs"] == 12
    assert bm["skipped_gpu_runs"] == 24
    assert bm["top_speedup_vs_cpu_ref"] == 44.60
    assert "MEAN vectorized" in bm["high_variability_label"]
    assert "34.29%" in bm["high_variability_label"]
    assert "SUM" not in bm["high_variability_label"]

    # Verify hardware summary
    hw = data["hardware"]
    assert hw["local_compiler_status"] == "UNAVAILABLE_LOCAL_COLAB_READY"
    assert "Tesla T4" in hw["canonical_colab_gpu"]

def test_dashboard_artifact_path_and_real_parsing():
    import os
    from backend.app.api.v1.endpoints.dashboard import WORKSPACE_ROOT, _load_evaluation_summary, _load_benchmark_summary

    # Verify WORKSPACE_ROOT points to the actual repository root containing experiments/
    experiments_dir = os.path.join(WORKSPACE_ROOT, "experiments")
    assert os.path.exists(experiments_dir), f"experiments/ not found under {WORKSPACE_ROOT}"

    eval_csv = os.path.join(WORKSPACE_ROOT, "experiments", "results", "evaluation_reporting_7_5", "summary_evaluation_metrics.csv")
    assert os.path.exists(eval_csv), f"Evaluation CSV missing at {eval_csv}"

    bench_csv = os.path.join(WORKSPACE_ROOT, "experiments", "results", "benchmark_runs", "session_20261024_230008_smoke", "summary_table.csv")
    assert os.path.exists(bench_csv), f"Benchmark CSV missing at {bench_csv}"

    # Verify real CSV parsing logic succeeds
    eval_summary = _load_evaluation_summary()
    assert eval_summary.top_test_pr_auc == 0.3315
    assert eval_summary.top_test_roc_auc == 0.7865
    assert eval_summary.top_test_f1 == 0.3265

    bench_summary = _load_benchmark_summary()
    assert bench_summary.total_runs == 36
    assert bench_summary.measured_cpu_runs == 12
    assert bench_summary.skipped_gpu_runs == 24

def test_list_datasets():
    response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_count"] == 4
    fixtures = {item["id"]: item for item in data["fixtures"]}
    assert "synthetic_fraud_10k" in fixtures
    assert "synthetic_tier_a_10k" in fixtures
    assert "synthetic_skewed_10k" in fixtures
    assert "toy_graph_5nodes" in fixtures

    # Verify authentic fixture values derived from actual files
    fraud_10k = fixtures["synthetic_fraud_10k"]
    assert fraud_10k["is_available"] is True
    assert fraud_10k["size_bytes"] == 1429459
    assert fraud_10k["num_nodes"] == 10000
    assert fraud_10k["num_edges"] == 231984
    assert fraud_10k["feature_dim"] == 32
    assert fraud_10k["sha256"] == "aecafe3b1fd608c18d62aa1586d0ab6f88b867bd92b20e0f6bd133606c7c4430"

    toy = fixtures["toy_graph_5nodes"]
    assert toy["is_available"] is True
    assert toy["size_bytes"] == 1406
    assert toy["num_nodes"] == 5
    assert toy["num_edges"] == 8
    assert toy["feature_dim"] == 4
    assert toy["sha256"] == "3ea8f702bc1f944d9f164222bad637666307c7be8e4a9767f813e40edc137909"

def test_list_datasets_missing_file_honest_representation(tmp_path):
    # Query datasets pointing to an empty temporary directory where no fixtures exist
    response = client.get(f"/api/v1/datasets?base_dir={tmp_path}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_count"] == 4

    for item in data["fixtures"]:
        # Unavailability must be honestly represented without hardcoded fallbacks
        assert item["is_available"] is False
        assert item["size_bytes"] == 0
        assert item["sha256"] == ""
        assert item["num_nodes"] == 0
        assert item["num_edges"] == 0
        assert item["feature_dim"] == 0

def test_list_datasets_corrupt_file_handling(tmp_path):
    # Create a corrupt fixture file (valid on filesystem, but unreadable as NPZ)
    corrupt_file = tmp_path / "toy_graph_5nodes.npz"
    corrupt_file.write_bytes(b"CORRUPTED_BINARY_CONTENT_NOT_AN_NPZ")

    response = client.get(f"/api/v1/datasets?base_dir={tmp_path}")
    assert response.status_code == 200
    data = response.json()
    fixtures = {item["id"]: item for item in data["fixtures"]}

    toy = fixtures["toy_graph_5nodes"]
    assert toy["is_available"] is False
    assert toy["size_bytes"] == len(b"CORRUPTED_BINARY_CONTENT_NOT_AN_NPZ")
    assert len(toy["sha256"]) == 64
    assert toy["num_nodes"] == 0
    assert toy["num_edges"] == 0
    assert toy["feature_dim"] == 0

def test_dashboard_evaluation_artifact_missing_handling(tmp_path):
    missing_eval_csv = tmp_path / "non_existent_eval.csv"
    response = client.get(f"/api/v1/dashboard/summary?eval_csv_path={missing_eval_csv}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"

    ev = data["evaluation"]
    assert ev["artifact_state"] == "MISSING"
    assert ev["top_test_pr_auc"] is None
    assert ev["top_test_roc_auc"] is None
    assert ev["top_test_f1"] is None
    assert ev["top_model"] is None
    assert "not found" in ev["error_message"].lower()

    # Valid benchmark artifact is preserved
    bm = data["benchmarks"]
    assert bm["artifact_state"] == "AVAILABLE"
    assert bm["top_speedup_vs_cpu_ref"] == 44.60

def test_dashboard_evaluation_artifact_invalid_empty_or_malformed(tmp_path):
    # Empty file
    empty_file = tmp_path / "empty_eval.csv"
    empty_file.write_text("", encoding="utf-8")

    response = client.get(f"/api/v1/dashboard/summary?eval_csv_path={empty_file}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["evaluation"]["artifact_state"] == "INVALID"
    assert data["evaluation"]["top_test_pr_auc"] is None
    assert "empty" in data["evaluation"]["error_message"].lower()

    # Missing required columns
    bad_cols_file = tmp_path / "bad_cols_eval.csv"
    bad_cols_file.write_text("col_a,col_b\n1,2\n", encoding="utf-8")

    response = client.get(f"/api/v1/dashboard/summary?eval_csv_path={bad_cols_file}")
    assert response.status_code == 200
    data = response.json()
    assert data["evaluation"]["artifact_state"] == "INVALID"
    assert data["evaluation"]["top_test_pr_auc"] is None
    assert "missing required columns" in data["evaluation"]["error_message"].lower()

def test_dashboard_evaluation_artifact_non_numeric_or_non_finite_rejection(tmp_path):
    # Non-numeric metric value
    bad_val_file = tmp_path / "bad_val_eval.csv"
    bad_val_file.write_text(
        "context_policy,model_name,representation,test_pr_auc,test_roc_auc,test_f1\n"
        "causal_history,logistic_regression,max,NOT_A_NUMBER,0.7865,0.3265\n",
        encoding="utf-8"
    )

    response = client.get(f"/api/v1/dashboard/summary?eval_csv_path={bad_val_file}")
    assert response.status_code == 200
    data = response.json()
    assert data["evaluation"]["artifact_state"] == "INVALID"
    assert data["evaluation"]["top_test_pr_auc"] is None
    assert "non-numeric" in data["evaluation"]["error_message"].lower()

    # Out of bounds probability metric (e.g. > 1.0)
    out_of_bounds_file = tmp_path / "out_of_bounds_eval.csv"
    out_of_bounds_file.write_text(
        "context_policy,model_name,representation,test_pr_auc,test_roc_auc,test_f1\n"
        "causal_history,logistic_regression,max,1.55,0.7865,0.3265\n",
        encoding="utf-8"
    )

    response = client.get(f"/api/v1/dashboard/summary?eval_csv_path={out_of_bounds_file}")
    assert response.status_code == 200
    data = response.json()
    assert data["evaluation"]["artifact_state"] == "INVALID"
    assert data["evaluation"]["top_test_pr_auc"] is None
    assert "interval [0.0, 1.0]" in data["evaluation"]["error_message"]

def test_dashboard_benchmark_artifact_missing_handling(tmp_path):
    missing_bench_csv = tmp_path / "non_existent_bench.csv"
    response = client.get(f"/api/v1/dashboard/summary?bench_csv_path={missing_bench_csv}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"

    bm = data["benchmarks"]
    assert bm["artifact_state"] == "MISSING"
    assert bm["top_speedup_vs_cpu_ref"] is None
    assert bm["top_speedup_workload"] is None
    assert bm["total_runs"] == 0
    assert "not found" in bm["error_message"].lower()

    # Valid evaluation artifact is preserved
    ev = data["evaluation"]
    assert ev["artifact_state"] == "AVAILABLE"
    assert ev["top_test_pr_auc"] == 0.3315

def test_dashboard_benchmark_artifact_invalid_or_malformed(tmp_path):
    # Empty benchmark CSV
    empty_bench = tmp_path / "empty_bench.csv"
    empty_bench.write_text("", encoding="utf-8")

    response = client.get(f"/api/v1/dashboard/summary?bench_csv_path={empty_bench}")
    assert response.status_code == 200
    data = response.json()
    assert data["benchmarks"]["artifact_state"] == "INVALID"
    assert data["benchmarks"]["top_speedup_vs_cpu_ref"] is None
    assert "empty" in data["benchmarks"]["error_message"].lower()

    # Missing speedup column
    bad_bench = tmp_path / "bad_bench.csv"
    bad_bench.write_text(
        "workload_name,operator,engine,measurement_state\n"
        "synthetic_tier_a_10k_mean,mean,cpu_vectorized,MEASURED\n",
        encoding="utf-8"
    )

    response = client.get(f"/api/v1/dashboard/summary?bench_csv_path={bad_bench}")
    assert response.status_code == 200
    data = response.json()
    assert data["benchmarks"]["artifact_state"] == "INVALID"
    assert data["benchmarks"]["top_speedup_vs_cpu_ref"] is None
    assert "missing required columns" in data["benchmarks"]["error_message"].lower()

def test_dashboard_path_traversal_guards():
    # Attempt to escape workspace with illegal path
    response = client.get("/api/v1/dashboard/summary?eval_csv_path=C:/Windows/System32/drivers/etc/hosts")
    assert response.status_code == 400
    assert "escapes authorized workspace boundaries" in response.json()["detail"]

def test_dashboard_evaluation_artifact_non_finite_and_negative_metrics(tmp_path):
    # Test nan, inf, -inf, and negative probability (-0.05)
    for bad_val in ["nan", "inf", "-inf", "-0.05"]:
        eval_csv = tmp_path / f"eval_{bad_val.replace('-', 'neg_')}.csv"
        eval_csv.write_text(
            "context_policy,model_name,representation,test_pr_auc,test_roc_auc,test_f1\n"
            f"causal_history,logistic_regression,max,{bad_val},0.7865,0.3265\n",
            encoding="utf-8"
        )
        response = client.get(f"/api/v1/dashboard/summary?eval_csv_path={eval_csv}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["evaluation"]["artifact_state"] == "INVALID"
        assert data["evaluation"]["top_test_pr_auc"] is None

def test_dashboard_benchmark_speedup_negative_zero_and_infinite(tmp_path):
    for bad_speedup in ["-5.0x", "0.0x", "infx"]:
        bench_csv = tmp_path / f"bench_{bad_speedup.replace('-', 'neg_')}.csv"
        bench_csv.write_text(
            "workload_name,operator,engine,measurement_state,speedup_vs_cpu_ref\n"
            f"synthetic_tier_a_10k_mean,mean,cpu_vectorized,MEASURED,{bad_speedup}\n",
            encoding="utf-8"
        )
        response = client.get(f"/api/v1/dashboard/summary?bench_csv_path={bench_csv}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["benchmarks"]["artifact_state"] == "INVALID"
        assert data["benchmarks"]["top_speedup_vs_cpu_ref"] is None
        assert "non-finite or non-positive speedup" in data["benchmarks"]["error_message"].lower()

def test_dashboard_evaluation_target_row_missing(tmp_path):
    eval_csv = tmp_path / "eval_missing_target_row.csv"
    # Headers are valid, rows exist, but no causal_history + logistic_regression + max
    eval_csv.write_text(
        "context_policy,model_name,representation,test_pr_auc,test_roc_auc,test_f1\n"
        "train_only_context,random_forest,mean,0.3120,0.7540,0.3010\n",
        encoding="utf-8"
    )
    response = client.get(f"/api/v1/dashboard/summary?eval_csv_path={eval_csv}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["evaluation"]["artifact_state"] == "INVALID"
    assert data["evaluation"]["top_test_pr_auc"] is None
    assert "target evaluation row" in data["evaluation"]["error_message"].lower()

def test_dashboard_benchmark_target_workload_row_missing(tmp_path):
    bench_csv = tmp_path / "bench_missing_target_row.csv"
    # Headers are valid, rows exist, but no synthetic_tier_a_10k_mean + cpu_vectorized
    bench_csv.write_text(
        "workload_name,operator,engine,measurement_state,speedup_vs_cpu_ref\n"
        "toy_graph_5nodes_max,max,cpu_reference,MEASURED,1.00x\n",
        encoding="utf-8"
    )
    response = client.get(f"/api/v1/dashboard/summary?bench_csv_path={bench_csv}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["benchmarks"]["artifact_state"] == "INVALID"
    assert data["benchmarks"]["top_speedup_vs_cpu_ref"] is None
    assert "target benchmark workload" in data["benchmarks"]["error_message"].lower()

def test_dashboard_relative_parent_path_traversal():
    response = client.get("/api/v1/dashboard/summary?eval_csv_path=../../outside.csv")
    assert response.status_code == 400
    assert "escapes authorized workspace boundaries" in response.json()["detail"]

    response_bench = client.get("/api/v1/dashboard/summary?bench_csv_path=../../outside.csv")
    assert response_bench.status_code == 400
    assert "escapes authorized workspace boundaries" in response_bench.json()["detail"]

def test_dashboard_variability_label_derives_operator_and_cv(tmp_path):
    # Case A: Selected row represents MEAN with CV 34.29%
    mean_bench_csv = tmp_path / "bench_mean_var.csv"
    mean_bench_csv.write_text(
        "workload_name,operator,engine,measurement_state,timing_stability,cv_percent,speedup_vs_cpu_ref\n"
        "synthetic_tier_a_10k_mean,mean,cpu_vectorized,MEASURED,HIGH_VARIABILITY,34.29,44.60x\n",
        encoding="utf-8"
    )
    res_mean = client.get(f"/api/v1/dashboard/summary?bench_csv_path={mean_bench_csv}")
    assert res_mean.status_code == 200
    bm_mean = res_mean.json()["benchmarks"]
    assert bm_mean["artifact_state"] == "AVAILABLE"
    assert "MEAN vectorized high variability documented (CV 34.29%)" in bm_mean["high_variability_label"]
    assert "SUM" not in bm_mean["high_variability_label"]

    # Case B: Selected row represents SUM with CV 189.0%
    sum_bench_csv = tmp_path / "bench_sum_var.csv"
    sum_bench_csv.write_text(
        "workload_name,operator,engine,measurement_state,timing_stability,cv_percent,speedup_vs_cpu_ref\n"
        "synthetic_tier_a_10k_mean,mean,cpu_vectorized,MEASURED,STABLE,10.0,44.60x\n"
        "synthetic_tier_a_10k_sum,sum,cpu_vectorized,MEASURED,HIGH_VARIABILITY,189.0,25.93x\n",
        encoding="utf-8"
    )
    res_sum = client.get(f"/api/v1/dashboard/summary?bench_csv_path={sum_bench_csv}")
    assert res_sum.status_code == 200
    bm_sum = res_sum.json()["benchmarks"]
    assert bm_sum["artifact_state"] == "AVAILABLE"
    assert "SUM vectorized high variability documented (CV 189.0%)" in bm_sum["high_variability_label"]

    # Case C: Missing operator in high variability row
    missing_op_csv = tmp_path / "bench_missing_op.csv"
    missing_op_csv.write_text(
        "workload_name,operator,engine,measurement_state,timing_stability,cv_percent,speedup_vs_cpu_ref\n"
        "synthetic_tier_a_10k_mean,,cpu_vectorized,MEASURED,HIGH_VARIABILITY,34.29,44.60x\n",
        encoding="utf-8"
    )
    res_missing_op = client.get(f"/api/v1/dashboard/summary?bench_csv_path={missing_op_csv}")
    assert res_missing_op.status_code == 200
    bm_missing_op = res_missing_op.json()["benchmarks"]
    assert "unavailable" in bm_missing_op["high_variability_label"].lower()

    # Case D: Missing/invalid CV in high variability row
    invalid_cv_csv = tmp_path / "bench_invalid_cv.csv"
    invalid_cv_csv.write_text(
        "workload_name,operator,engine,measurement_state,timing_stability,cv_percent,speedup_vs_cpu_ref\n"
        "synthetic_tier_a_10k_mean,mean,cpu_vectorized,MEASURED,HIGH_VARIABILITY,NOT_A_NUMBER,44.60x\n",
        encoding="utf-8"
    )
    res_invalid_cv = client.get(f"/api/v1/dashboard/summary?bench_csv_path={invalid_cv_csv}")
    assert res_invalid_cv.status_code == 200
    bm_invalid_cv = res_invalid_cv.json()["benchmarks"]
    assert "unavailable" in bm_invalid_cv["high_variability_label"].lower()
