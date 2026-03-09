from pyfolds.optimization.compile_benchmark import run_compile_benchmark


def test_compile_benchmark_runs(tmp_path):
    out = tmp_path / "bench.json"
    results = run_compile_benchmark(str(out))
    assert "eager" in results
    assert out.exists()
