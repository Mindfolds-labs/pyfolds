from __future__ import annotations

from train_mnist import parse_args, run_training


if __name__ == "__main__":
    args = parse_args()
    args.backend = "mind"
    args.save_fold = False
    args.save_mind = True
    raise SystemExit(run_training(args))
