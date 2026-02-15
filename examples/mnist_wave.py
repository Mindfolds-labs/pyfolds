"""Esboço: treinamento MNIST com MPJRDWaveNeuron e codificação de frequência."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pyfolds.wave import MPJRDWaveConfig, MPJRDWaveNeuron


def to_dendritic_input(img: torch.Tensor, cfg: MPJRDWaveConfig) -> torch.Tensor:
    flat = img.view(-1)
    needed = cfg.n_dendrites * cfg.n_synapses_per_dendrite
    if flat.numel() < needed:
        flat = torch.nn.functional.pad(flat, (0, needed - flat.numel()))
    flat = flat[:needed]
    return flat.view(1, cfg.n_dendrites, cfg.n_synapses_per_dendrite)


def main():
    cfg = MPJRDWaveConfig(
        n_dendrites=8,
        n_synapses_per_dendrite=98,
        theta_init=4.0,
        base_frequency=12.0,
        frequency_step=4.0,
        class_frequencies=tuple(12 + 4 * i for i in range(10)),
    )
    neuron = MPJRDWaveNeuron(cfg)

    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    for step, (img, label) in enumerate(loader):
        x = to_dendritic_input(img[0], cfg)
        out = neuron(x, reward=1.0, target_class=int(label.item()))

        # recompensa simples: reforça quando há spike com boa sincronia
        reward = float(out["spikes"].item()) * float(out["phase_sync"].mean().item())
        neuron.apply_plasticity(dt=1.0, reward=reward)

        if step % 200 == 0:
            print(
                f"step={step} label={label.item()} spike={out['spikes'].item():.0f} "
                f"u={out['u'].item():.3f} freq={out['frequency'].item():.1f} "
                f"phase={out['phase'].item():.3f}"
            )

        if step >= 1000:
            break


if __name__ == "__main__":
    main()
