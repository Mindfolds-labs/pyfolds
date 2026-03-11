from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Importação dos mecanismos avançados do PyFolds
try:
    from pyfolds.advanced import MPJRDNeuronAdvanced
    from pyfolds.core.config import MPJRDConfig
    from pyfolds.utils.types import LearningMode
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pyfolds.advanced import MPJRDNeuronAdvanced
    from pyfolds.core.config import MPJRDConfig
    from pyfolds.utils.types import LearningMode

from serialization.folds_io import save_model_fold
from serialization.mind_io import save_model_mind

try:
    import torchvision
    import torchvision.transforms as transforms
except Exception:
    torchvision = None
    transforms = None


@dataclass
class TrainArgs:
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

    # Parâmetros de arquitetura MPJRD
    n_dendrites: int = 4
    n_synapses_per_dendrite: int = 32
    hidden: int = 128
    threshold: float = 0.45

    # Controle de formatos de saída
    save_fold: int = 1
    save_mind: int = 1
    save_pt: int = 1
    save_log: int = 1
    save_metrics: int = 1
    save_summary: int = 1

    # Controle de mecanismos (desabilitar)
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


# =============================================================================
# LAYOUT DE INICIALIZAÇÃO COM CAIXAS UNICODE
# =============================================================================
def _format_line(content: str, width: int, align: str = "left") -> str:
    """Formata uma linha para caber dentro da caixa com largura fixa."""
    if align == "left":
        padded = content.ljust(width)
    elif align == "center":
        padded = content.center(width)
    elif align == "right":
        padded = content.rjust(width)
    else:
        padded = content.ljust(width)
    return f"║ {padded} ║"


def _print_box(title: str, lines: list[str], width: int) -> None:
    """Imprime uma caixa com título e linhas de conteúdo."""
    print(f"╔{'═' * (width + 2)}╗")
    if title:
        print(_format_line(title, width, "center"))
        print(f"╠{'═' * (width + 2)}╣")
    for line in lines:
        print(_format_line(line, width, "left"))
    print(f"╚{'═' * (width + 2)}╝")


def print_experiment_layout(args: TrainArgs, cfg: MPJRDConfig) -> None:
    """Imprime o layout completo do experimento com alinhamento absoluto."""
    if not args.console:
        return

    import shutil

    term_width = shutil.get_terminal_size().columns
    box_width = min(78, term_width - 4)

    # ===== CABEÇALHO =====
    header_lines = [
        f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}      📁 Run ID: {args.run_id}",
        f"🏗️ Framework: PyFolds v2.1.1      ⚙️ Backend: PyTorch {torch.__version__}",
        f"💻 Dispositivo: {args.device.upper()}              🎯 Modo: ONLINE (aprendizado contínuo)",
    ]
    _print_box("", header_lines, box_width)
    print()

    # ===== CONFIGURAÇÃO DO MODELO =====
    model_lines = [
        "Modelo base        : MPJRD",
        f"Dendritos          : {cfg.n_dendrites}",
        f"Sinapses/dendrito  : {cfg.n_synapses_per_dendrite}",
        f"Integração         : {cfg.dendrite_integration_mode} (gain={cfg.dendrite_gain}, θ_ratio={cfg.theta_dend_ratio})",
        f"Hidden dim         : {args.hidden} (neurônios excitatórios)",
        f"Threshold inicial  : {cfg.theta_init}",
        f"Learning rate      : {args.lr}",
        f"Batch size         : {args.batch}",
    ]
    _print_box("CONFIGURAÇÃO DO MODELO", model_lines, box_width)
    print()

    # ===== MECANISMOS BIOLÓGICOS =====
    mech_lines: list[str] = []

    mech_lines.append("[OBRIGATÓRIOS]")
    mech_lines.append(f"  Filamentos (N)      ✅  n_min={cfg.n_min}, n_max={cfg.n_max}, w_scale={cfg.w_scale}")
    mech_lines.append(f"  Potencial interno (I)✅  i_min={cfg.i_min}, i_max={cfg.i_max}, i_eta={cfg.i_eta}")
    mech_lines.append("")

    stp_status = "✅" if not args.disable_stp else "❌"
    mech_lines.append("[PLASTICIDADE DE CURTO PRAZO]")
    mech_lines.append(f"  STP                  {stp_status}  u0={cfg.u0}, R0={cfg.R0}, U={cfg.U}, τ_fac={cfg.tau_fac}ms,")
    mech_lines.append(f"                                  τ_rec={cfg.tau_rec}ms")
    mech_lines.append("")

    stdp_status = "✅" if not args.disable_stdp else "❌"
    backprop_status = "✅" if not args.disable_backprop else "❌"
    mech_lines.append("[PLASTICIDADE DE LONGO PRAZO]")
    mech_lines.append(f"  STDP                 {stdp_status}  modo={cfg.plasticity_mode}, A⁺={cfg.A_plus}, A⁻={cfg.A_minus},")
    mech_lines.append(f"                                  τ_pre={cfg.tau_pre}ms, τ_post={cfg.tau_post}ms")
    mech_lines.append(f"  Backpropagação       {backprop_status}  delay={cfg.backprop_delay}ms, sinal={cfg.backprop_signal}, proporcional={cfg.bap_proportional}")
    mech_lines.append(f"  Hebbian eligibility  ✅  β_w={cfg.beta_w}, LTD_ratio={cfg.hebbian_ltd_ratio}")
    mech_lines.append("")

    homeo_status = "✅" if not args.disable_homeostase else "❌"
    sfa_status = "✅" if not args.disable_sfa else "❌"
    mech_lines.append("[CONTROLE HOMEOSTÁTICO]")
    mech_lines.append(f"  Homeostase           {homeo_status}  θ_alvo={cfg.target_spike_rate}, η={cfg.homeostasis_eta}, θ∈[{cfg.theta_min}, {cfg.theta_max}]")
    mech_lines.append(f"  Adaptação SFA        {sfa_status}  inc={cfg.adaptation_increment}, τ={cfg.adaptation_tau}ms, max={cfg.adaptation_max}")
    mech_lines.append("")

    ref_status = "✅" if not args.disable_refratario else "❌"
    inh_status = "✅" if not args.disable_inibicao else "❌"
    mech_lines.append("[DINÂMICA DE DISPARO]")
    mech_lines.append(f"  Refratário           {ref_status}  abs={cfg.t_refrac_abs}ms, rel={cfg.t_refrac_rel}ms, boost={cfg.refrac_rel_strength}")
    mech_lines.append(f"  Inibição             {inh_status}  lateral={cfg.lateral_strength}, feedback={cfg.feedback_strength}, E={cfg.n_excitatory}, I={cfg.n_inhibitory}")
    mech_lines.append("")

    wave_status = "✅" if cfg.wave_enabled and not args.disable_wave else "❌"
    circ_status = "✅" if cfg.circadian_enabled and not args.disable_circadian else "❌"
    engram_status = "✅" if cfg.experimental_engram_enabled and not args.disable_engram else "❌"
    speech_status = "✅" if cfg.enable_speech_envelope_tracking and not args.disable_speech else "❌"
    mech_lines.append("[OPCIONAIS / EXPERIMENTAIS]")
    mech_lines.append(f"  Wave oscilatório     {wave_status}  (n_freq={cfg.wave_n_frequencies}, base={cfg.wave_base_frequency}Hz, sleep_consolidation={cfg.wave_sleep_consolidation})")
    mech_lines.append(f"  Circadiano           {circ_status}  (ciclo={cfg.circadian_cycle_hours}h, auto_mode={cfg.circadian_auto_mode})")
    mech_lines.append(f"  Memória (Engrams)    {engram_status}  max_engrams={cfg.max_engrams}, pruning_th={cfg.pruning_threshold},")
    mech_lines.append(f"                                  fase_resonance={cfg.enable_experimental_phase_resonance}")
    mech_lines.append(f"  Speech tracking      {speech_status}  (método={cfg.speech_envelope_method})")

    _print_box("MECANISMOS BIOLÓGICOS", mech_lines, box_width)
    print()

    fmt_lines = [
        f"[{'✅' if args.save_fold else '❌'}] .fold   [{'✅' if args.save_mind else '❌'}] .mind   "
        f"[{'✅' if args.save_pt else '❌'}] .pt   [{'✅' if args.save_metrics else '❌'}] .jsonl  "
        f"[{'✅' if args.save_summary else '❌'}] .json"
    ]
    _print_box("FORMATOS DE SAÍDA", fmt_lines, box_width)
    print()

    total_synapses = cfg.n_dendrites * cfg.n_synapses_per_dendrite
    vc_approx = total_synapses * math.log2(total_synapses + 1)
    trainable_params = total_synapses
    cap_lines = [
        f"Sinapses totais         : {total_synapses}",
        f"VC-dimension aproximada : {vc_approx:.1e}",
        f"Parâmetros treináveis   : {trainable_params:,}".replace(",", "."),
    ]
    _print_box("MÉTRICAS DE CAPACIDADE", cap_lines, box_width)
    print()


class MPJRDWrapper(nn.Module):
    """Wrapper para adaptar MNIST [B,1,28,28] para entrada MPJRD [B,D,S]."""

    def __init__(self, neuron: MPJRDNeuronAdvanced, cfg: MPJRDConfig):
        super().__init__()
        self.neuron = neuron
        self.cfg = cfg

        self.proj = nn.Linear(784, cfg.n_dendrites * cfg.n_synapses_per_dendrite)
        self.classifier = nn.Linear(cfg.n_dendrites * cfg.n_synapses_per_dendrite, 10)
        nn.init.xavier_uniform_(self.proj.weight, gain=2.0)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        bsz = x.shape[0]
        x_flat = x.view(bsz, -1)
        x_proj = self.proj(x_flat)
        x_reshaped = x_proj.view(bsz, self.cfg.n_dendrites, self.cfg.n_synapses_per_dendrite)
        neuron_out = self.neuron(x_reshaped, **kwargs)

        if isinstance(neuron_out, dict):
            features = neuron_out.get("spikes", neuron_out.get("u", x_reshaped.mean(dim=(1, 2))))
        else:
            features = x_reshaped.mean(dim=(1, 2))

        if features.dim() > 2:
            features = features.view(bsz, -1)
        elif features.dim() == 1:
            features = features.unsqueeze(1)

        if features.shape[1] != self.cfg.n_dendrites * self.cfg.n_synapses_per_dendrite:
            features = x_reshaped.view(bsz, -1)

        logits = self.classifier(features)
        return logits, neuron_out

    def get_config(self):
        return self.neuron.get_config() if hasattr(self.neuron, "get_config") else {}

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.neuron, name)


def _setup_logger(run_dir: Path, log_file: str, console: bool) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"mnist.{run_dir.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(message)s")

    f_handler = logging.FileHandler(run_dir / log_file, encoding="utf-8")
    f_handler.setFormatter(fmt)
    logger.addHandler(f_handler)

    if console:
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(fmt)
        logger.addHandler(c_handler)

    return logger


def _build_loaders(batch_size: int):
    if torchvision is not None and transforms is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        try:
            train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
            return train_loader, test_loader
        except Exception:
            pass

    x_train = torch.rand(2048, 1, 28, 28)
    y_train = torch.randint(0, 10, (2048,))
    x_test = torch.rand(512, 1, 28, 28)
    y_test = torch.randint(0, 10, (512,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=512, shuffle=False)
    return train_loader, test_loader


def _save_backend(backend: str, path: Path, payload: dict[str, Any]) -> None:
    if backend == "folds":
        save_model_fold(path, payload)
    else:
        save_model_mind(path, payload)


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

        raw_neuron = MPJRDNeuronAdvanced(cfg=cfg)
        model = MPJRDWrapper(raw_neuron, cfg).to(device)

        print_experiment_layout(args, cfg)

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
                    out = model(x, mode=LearningMode.ONLINE)
                    logits = out[0] if isinstance(out, tuple) else out
                    if isinstance(logits, dict):
                        logits = logits.get("logits", logits.get("spikes", torch.zeros(x.size(0), 10, device=device)))
                    loss = criterion(logits, y)

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    total_loss += loss.item()
                    pred = logits.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)

                    if isinstance(out, dict):
                        spike_rate = out.get("spike_rate", 0.0)
                        if spike_rate:
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
                        out = model(x, mode=LearningMode.INFERENCE)
                        logits = out[0] if isinstance(out, tuple) else out
                        if isinstance(logits, dict):
                            logits = logits.get("logits", logits.get("spikes", torch.zeros(x.size(0), 10, device=device)))
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

                if hasattr(model, "sleep") and not args.disable_homeostase:
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
                "model_type": "mpjrd",
                "run_id": args.run_id,
                "timestamp": datetime.now().isoformat(),
                "hyperparameters": {
                    "batch_size": args.batch,
                    "learning_rate": args.lr,
                    "epochs": args.epochs,
                    "device": args.device,
                    "model": "mpjrd",
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
                "model": "mpjrd",
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
                "model": "mpjrd",
                "status": "failed",
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            }
            (run_dir / "summary.json").write_text(json.dumps(fail_summary, indent=2))
        return 1
