from __future__ import annotations

from train_mnist import parse_args, run_training


if __name__ == "__main__":
    args = parse_args()
    args.backend = "folds"
    args.save_fold = True
    args.save_mind = False
    raise SystemExit(run_training(args))
