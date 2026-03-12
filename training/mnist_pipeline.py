from __future__ import annotations

from dataclasses import dataclass

from training.config.mnist import BaseTrainConfig, FOLDSNetTrainConfig, MPJRDTrainConfig, RunConfig
from training.trainers.mnist_trainer import run_mnist_training


@dataclass
class TrainArgs:
    """Legacy args mantidos para compatibilidade dos scripts existentes."""

    backend: str
    epochs: int
    batch: int
    lr: float
    run_id: str
    resume: bool
    device: str
    console: bool
    log_level: str
    log_file: str
    sheer_cmd: str = ""
    model: str = "mpjrd"
    timesteps: int = 4

    n_dendrites: int = 4
    n_synapses_per_dendrite: int = 32
    hidden: int = 128
    threshold: float = 0.45

    save_fold: int = 1
    save_mind: int = 1
    save_pt: int = 1
    save_log: int = 1
    save_metrics: int = 1
    save_summary: int = 1

    disable_stdp: bool = False
    disable_homeostase: bool = False
    disable_inibicao: bool = False
    disable_refratario: bool = False
    disable_backprop: bool = False
    disable_sfa: bool = False
    disable_stp: bool = False
    disable_wave: bool = False
    disable_circadian: bool = False
    disable_engram: bool = False
    disable_speech: bool = False

    foldsnet_variant: str = "4L"
    foldsnet_dataset: str = "mnist"


def _to_run_config(args: TrainArgs) -> RunConfig:
    base = BaseTrainConfig(
        backend=args.backend,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        run_id=args.run_id,
        resume=args.resume,
        device=args.device,
        console=args.console,
        log_level=args.log_level,
        log_file=args.log_file,
        sheer_cmd=args.sheer_cmd,
        timesteps=args.timesteps,
        save_fold=args.save_fold,
        save_mind=args.save_mind,
        save_pt=args.save_pt,
        save_log=args.save_log,
        save_metrics=args.save_metrics,
        save_summary=args.save_summary,
    )
    mpjrd = MPJRDTrainConfig(
        n_dendrites=args.n_dendrites,
        n_synapses_per_dendrite=args.n_synapses_per_dendrite,
        hidden=args.hidden,
        threshold=args.threshold,
        disable_stdp=args.disable_stdp,
        disable_homeostase=args.disable_homeostase,
        disable_inibicao=args.disable_inibicao,
        disable_refratario=args.disable_refratario,
        disable_backprop=args.disable_backprop,
        disable_sfa=args.disable_sfa,
        disable_stp=args.disable_stp,
        disable_wave=args.disable_wave,
        disable_circadian=args.disable_circadian,
        disable_engram=args.disable_engram,
        disable_speech=args.disable_speech,
    )
    foldsnet = FOLDSNetTrainConfig(variant=args.foldsnet_variant, dataset=args.foldsnet_dataset)
    return RunConfig(base=base, mpjrd=mpjrd, foldsnet=foldsnet)


def _extract_spike_rate(output: Any) -> float:
    """Extrai taxa de spike de saídas heterogêneas (tensor, dict ou tuple)."""
    candidate: Any = output
    if isinstance(output, tuple) and len(output) >= 2:
        candidate = output[1]

    if not isinstance(candidate, dict):
        return 0.0

    spike_rate = candidate.get("spike_rate")
    if spike_rate is not None:
        if torch.is_tensor(spike_rate):
            return float(spike_rate.detach().float().mean().item())
        try:
            return float(spike_rate)
        except (TypeError, ValueError):
            return 0.0

    spikes = candidate.get("spikes")
    if torch.is_tensor(spikes):
        return float(spikes.detach().float().mean().item())
    return 0.0


def run_training(args: TrainArgs) -> int:
    run_dir = Path("runs") / args.run_id
    logger = _setup_logger(run_dir, args.log_file, args.console)
    device = torch.device(args.device)

    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    artifact_path = run_dir / ("model.fold" if args.backend == "folds" else "model.mind")

    test_acc = 0.0
    epoch_loss = 0.0
    epochs_completed = 0
    best_acc = 0.0

    try:
        logging.getLogger("pyfolds").setLevel(logging.WARNING)
        logging.getLogger("pyfolds.advanced").setLevel(logging.WARNING)
        logging.getLogger("pyfolds.core").setLevel(logging.WARNING)
        logging.getLogger("pyfolds.advanced.dendrite").setLevel(logging.ERROR)
        logging.getLogger("pyfolds.advanced.homeostasis").setLevel(logging.ERROR)
        logging.getLogger("pyfolds.advanced.neuron").setLevel(logging.ERROR)
        logging.getLogger("pyfolds.core.inhibition").setLevel(logging.ERROR)

        cfg = MPJRDConfig(
            n_dendrites=args.n_dendrites,
            n_synapses_per_dendrite=args.n_synapses_per_dendrite,
            n_min=1,
            n_max=100,
            w_scale=3.5,
            i_eta=args.lr,
            i_gamma=0.99,
            i_min=-20.0,
            i_max=50.0,
            u0=0.2,
            R0=1.0,
            U=0.3,
            tau_fac=100.0,
            tau_rec=800.0,
            theta_init=args.threshold,
            theta_min=0.2,
            theta_max=4.0,
            target_spike_rate=0.2,
            homeostasis_eta=0.2,
            dead_neuron_threshold=0.01,
            activity_threshold=0.01,
            dendrite_integration_mode="nmda_shunting",
            dendrite_gain=2.0,
            backprop_enabled=not args.disable_backprop,
            backprop_delay=2.0,
            backprop_signal=0.5,
            adaptation_enabled=not args.disable_sfa,
            adaptation_increment=0.8,
            adaptation_decay=0.99,
            adaptation_max=5.0,
            adaptation_tau=50.0,
            refrac_mode="both" if not args.disable_refratario else "none",
            t_refrac_abs=2.0,
            t_refrac_rel=5.0,
            refrac_rel_strength=3.0,
            inhibition_mode="both" if not args.disable_inibicao else "none",
            lateral_strength=0.3,
            feedback_strength=0.5,
            n_excitatory=args.hidden,
            n_inhibitory=args.hidden // 4,
            plasticity_mode="both" if not args.disable_stdp else "none",
            tau_pre=20.0,
            tau_post=20.0,
            A_plus=0.01,
            A_minus=0.012,
            neuromod_mode="surprise",
            sup_k=3.0,
            wave_enabled=not args.disable_wave,
            circadian_enabled=not args.disable_circadian,
            experimental_engram_enabled=not args.disable_engram,
            enable_speech_envelope_tracking=not args.disable_speech,
            plastic=True,
            defer_updates=True,
            dt=1.0,
            device=str(device),
        )

        if args.model == "mpjrd":
            raw_neuron = MPJRDNeuronAdvanced(cfg=cfg)
            model = MPJRDWrapper(raw_neuron, cfg).to(device)
            print_experiment_layout(args, cfg)
        elif args.model == "foldsnet":
            if importlib.util.find_spec("foldsnet") is None:
                root_path = Path(__file__).resolve().parents[1]
                if str(root_path) not in sys.path:
                    sys.path.insert(0, str(root_path))
            from foldsnet.factory import create_foldsnet

            model = create_foldsnet("4L", "mnist").to(device)
            logger.info("🧠 Modo FOLDSNet ativo: mecanismos MPJRD por-camada não aplicados neste pipeline.")
            if args.console:
                print("🧠 Modo FOLDSNet ativo: mecanismos avançados do pipeline MPJRD estão inativos.")
        else:
            raise ValueError("Modelo inválido. Use 'mpjrd' ou 'foldsnet'.")

        with torch.no_grad():
            _ = model(torch.zeros(1, 1, 28, 28, device=device))

        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        if args.resume and checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optim.load_state_dict(ckpt["optimizer_state"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_acc = float(ckpt.get("best_acc", 0.0))
            logger.info(f"🔄 RESUMO epoch={start_epoch}")

        train_loader, test_loader = _build_loaders(args.batch)
        epochs_completed = start_epoch

        with metrics_path.open("a", encoding="utf-8") as mf:
            for epoch in range(start_epoch, args.epochs):
                model.train()
                if hasattr(model, "set_mode"):
                    model.set_mode(LearningMode.ONLINE)

                total_loss = 0.0
                correct = 0
                total = 0
                spike_rates: list[float] = []

                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    if args.model == "mpjrd":
                        out = model(x, mode=LearningMode.ONLINE)
                    else:
                        out = model(x)
                    logits = _extract_logits(out, x.size(0), device)
                    loss = criterion(logits, y)

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    total_loss += loss.item()
                    pred = logits.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)

                    spike_rate = _extract_spike_rate(out)
                    spike_rates.append(spike_rate)

                train_acc = 100.0 * correct / total
                avg_loss = total_loss / len(train_loader)
                avg_spike = sum(spike_rates) / len(spike_rates) if spike_rates else 0.0

                model.eval()
                if hasattr(model, "set_mode"):
                    model.set_mode(LearningMode.INFERENCE)

                t_correct, t_total = 0, 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        if args.model == "mpjrd":
                            out = model(x, mode=LearningMode.INFERENCE)
                        else:
                            out = model(x)
                        logits = _extract_logits(out, x.size(0), device)
                        pred = logits.argmax(dim=1)
                        t_correct += (pred == y).sum().item()
                        t_total += y.size(0)

                test_acc = 100.0 * t_correct / t_total
                best_acc = max(best_acc, test_acc)
                epoch_loss = avg_loss

                if args.console:
                    print(
                        f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | "
                        f"train={train_acc:.2f}% | test={test_acc:.2f}% | spike={avg_spike:.6f}"
                    )

                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | "
                    f"train={train_acc:.2f}% | test={test_acc:.2f}% | spike={avg_spike:.6f}"
                )

                if args.save_metrics:
                    mf.write(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "loss": avg_loss,
                                "train_acc_pct": train_acc,
                                "test_acc_pct": test_acc,
                                "spike_rate": avg_spike,
                            }
                        )
                        + "\n"
                    )
                    mf.flush()

                if args.model == "mpjrd" and hasattr(model, "sleep") and not args.disable_homeostase:
                    logger.info("💤 Ciclo de sono (consolidação I → N)")
                    model.sleep(duration=60.0)

                if args.save_pt:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optim.state_dict(),
                            "best_acc": best_acc,
                        },
                        checkpoint_path,
                    )

                epochs_completed = epoch + 1

        if args.console:
            print(f"🏆 FINAL SUMMARY | final_acc={test_acc:.2f}% | best_acc={best_acc:.2f}% | final_loss={epoch_loss:.4f}")

        logger.info(f"FINAL SUMMARY | final_acc={test_acc:.2f}% | best_acc={best_acc:.2f}% | final_loss={epoch_loss:.4f}")

        if args.save_fold or args.save_mind:
            payload = {
                "model_state": model.state_dict(),
                "model_config": model.get_config() if hasattr(model, "get_config") else {},
                "epoch": args.epochs - 1,
                "best_acc": best_acc,
                "final_acc": test_acc,
                "final_loss": epoch_loss,
                "model_type": args.model,
                "run_id": args.run_id,
                "timestamp": datetime.now().isoformat(),
                "hyperparameters": {
                    "batch_size": args.batch,
                    "learning_rate": args.lr,
                    "epochs": args.epochs,
                    "device": args.device,
                    "model": args.model,
                    "n_dendrites": args.n_dendrites,
                    "n_synapses_per_dendrite": args.n_synapses_per_dendrite,
                    "hidden": args.hidden,
                    "threshold": args.threshold,
                },
            }

            if args.save_fold:
                _save_backend("folds", artifact_path, payload)
                if args.console:
                    print(f"💾 Modelo .fold salvo em: {artifact_path}")

            if args.save_mind:
                mind_path = run_dir / "model.mind"
                _save_backend("mind", mind_path, payload)
                if args.console:
                    print(f"🧠 Modelo .mind salvo em: {mind_path}")

        if args.save_summary:
            summary = {
                "run_id": args.run_id,
                "backend": args.backend,
                "model": args.model,
                "epochs_requested": args.epochs,
                "epochs_completed": epochs_completed,
                "resume_used": args.resume,
                "best_acc_pct": best_acc,
                "final_acc_pct": test_acc,
                "final_loss": epoch_loss,
                "model_config": model.get_config() if hasattr(model, "get_config") else {},
                "train_config": asdict(args),
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return 0

    except Exception as exc:
        logger.exception("❌ FALHA NO TREINO")
        if args.console:
            print(f"❌ FALHA NO TREINO: {exc}")
        if args.save_summary:
            fail_summary = {
                "run_id": args.run_id,
                "model": args.model,
                "status": "failed",
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            }
            (run_dir / "summary.json").write_text(json.dumps(fail_summary, indent=2))
        return 1
