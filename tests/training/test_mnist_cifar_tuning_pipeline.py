from training.mnist_cifar_tuning_pipeline import run_tuning


def test_run_tuning_simulation_smoke() -> None:
    report = run_tuning(
        datasets=["mnist", "cifar10"],
        epochs=1,
        data_root="./.data",
        simulate=True,
        train_subset=128,
        test_subset=64,
        num_workers=0,
        device="cpu",
        top_k=2,
    )

    assert "datasets" in report
    assert "mnist" in report["datasets"]
    assert "cifar10" in report["datasets"]

    mnist_best = report["datasets"]["mnist"]["best"]
    cifar_best = report["datasets"]["cifar10"]["best"]

    assert mnist_best["steps_per_epoch"] > 0
    assert cifar_best["steps_per_epoch"] > 0
    assert 0.0 <= mnist_best["test_acc_pct"] <= 100.0
    assert 0.0 <= cifar_best["test_acc_pct"] <= 100.0
