"""CNN vs SNN (snnTorch) on MNIST: models, training, op counting, latency, FGSM."""
import random
import subprocess

import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from snntorch import surrogate
from thop import profile
from torchvision import datasets

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Horowitz, ISSCC 2014, 45 nm, 32-bit float: mult 3.7 pJ + add 0.9 pJ = 4.6 pJ/MAC; add 0.9 pJ/AC.
E_MAC_PJ, E_AC_PJ = 4.6, 0.9
# Hidden + output LIF neurons per sample: 12x12x12 + 64x4x4 + 128 + 10
N_NEURONS = 12 * 12 * 12 + 64 * 4 * 4 + 128 + 10
CNN_MACS_ANALYTIC = 24 * 24 * 12 * 25 + 8 * 8 * 64 * 12 * 25 + 1024 * 128 + 128 * 10


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mnist(root="data"):
    """Whole MNIST as GPU tensors in [0, 1] (no normalisation, so rate coding and FGSM eps share one scale)."""
    out = []
    for train in (True, False):
        ds = datasets.MNIST(root, train=train, download=True)
        out += [ds.data.float().div(255).unsqueeze(1).to(DEV), ds.targets.to(DEV)]
    return out  # xtr, ytr, xte, yte


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(1, 12, 5), nn.Conv2d(12, 64, 5)
        self.fc1, self.fc2 = nn.Linear(1024, 128), nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def conv_sops(spikes, conv):
    """Exact count of synaptic accumulates: every (input spike, output neuron) connection (stride 1, no padding)."""
    k = torch.ones(1, conv.in_channels, *conv.kernel_size, device=spikes.device)
    return F.conv2d(spikes, k).sum() * conv.out_channels


class SNN(nn.Module):
    """Same layers as CNN; each ReLU replaced by a Leaky (LIF) neuron. Output = spike counts over T."""

    # beta=0.9: with beta=0.5 the T=5 net stayed silent and did not learn (0.098 acc after 1 epoch, see README).
    def __init__(self, T, beta=0.9, slope=25):
        super().__init__()
        self.T = T
        self.conv1, self.conv2 = nn.Conv2d(1, 12, 5), nn.Conv2d(12, 64, 5)
        self.fc1, self.fc2 = nn.Linear(1024, 128), nn.Linear(128, 10)
        sg = surrogate.fast_sigmoid(slope=slope)
        self.lif = nn.ModuleList(snn.Leaky(beta=beta, spike_grad=sg) for _ in range(4))

    def forward(self, x, count=False):
        mem = [l.init_leaky() for l in self.lif]
        out = 0
        sops = spikes = torch.zeros((), device=x.device)
        for _ in range(self.T):
            # Rate coding: Bernoulli(pixel) per step (what snntorch.spikegen.rate does), with a
            # straight-through gradient so FGSM can reach the pixels.
            inp = x + (torch.bernoulli(x) - x).detach()
            s1, mem[0] = self.lif[0](F.max_pool2d(self.conv1(inp), 2), mem[0])
            s2, mem[1] = self.lif[1](F.max_pool2d(self.conv2(s1), 2), mem[1])
            s3, mem[2] = self.lif[2](self.fc1(s2.flatten(1)), mem[2])
            s4, mem[3] = self.lif[3](self.fc2(s3), mem[3])
            out = out + s4
            if count:
                sops = sops + conv_sops(inp, self.conv1) + conv_sops(s1, self.conv2) + s2.sum() * 128 + s3.sum() * 10
                spikes = spikes + s1.sum() + s2.sum() + s3.sum() + s4.sum()
        return (out, sops, spikes) if count else out


def cnn_macs():
    m = CNN().to(DEV)
    macs, _ = profile(m, inputs=(torch.zeros(1, 1, 28, 28, device=DEV),), verbose=False)
    return int(macs)


def self_check():
    """SOP counter on an all-ones input must equal the dense MAC count; thop must match the analytic MACs."""
    conv = nn.Conv2d(1, 12, 5).to(DEV)
    assert conv_sops(torch.ones(1, 1, 28, 28, device=DEV), conv).item() == 24 * 24 * 12 * 25
    macs = cnn_macs()
    assert macs == CNN_MACS_ANALYTIC, (macs, CNN_MACS_ANALYTIC)
    return macs


def train(model, x, y, epochs, bs=128, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=x.device)
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    """Returns accuracy, SOPs per inference, spikes per neuron per inference (last two None for CNN)."""
    model.eval()
    correct, sops, spikes = 0, 0.0, 0.0
    for i in range(0, len(x), bs):
        xb, yb = x[i:i + bs], y[i:i + bs]
        if isinstance(model, SNN):
            out, so, sp = model(xb, count=True)
            sops += so.item()
            spikes += sp.item()
        else:
            out = model(xb)
        correct += (out.argmax(1) == yb).sum().item()
    n = len(x)
    if isinstance(model, SNN):
        return correct / n, sops / n, spikes / (n * N_NEURONS)
    return correct / n, None, None


def fgsm_acc(model, x, y, eps, bs=500):
    """White-box FGSM in [0,1] pixel space. For the SNN, gradients flow through fast_sigmoid surrogates."""
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        xb, yb = x[i:i + bs].clone().requires_grad_(True), y[i:i + bs]
        (g,) = torch.autograd.grad(F.cross_entropy(model(xb), yb), xb)
        xa = (xb + eps * g.sign()).clamp(0, 1).detach()
        with torch.no_grad():
            correct += (model(xa).argmax(1) == yb).sum().item()
    return correct / len(x)


@torch.no_grad()
def latency_ms(model, x, warmup, iters):
    """Forward-pass wall-clock per batch (CUDA events), after warm-up. Returns mean, std over timed runs."""
    model.eval()
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        model(x)
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return float(np.mean(times)), float(np.std(times))


def gpu_state():
    keys = ["temp_c", "sm_clock_mhz", "sm_clock_max_mhz", "power_w", "throttle_reasons"]
    q = "temperature.gpu,clocks.sm,clocks.max.sm,power.draw,clocks_throttle_reasons.active"
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    d = dict(zip(keys, (v.strip() for v in r.stdout.strip().split(","))))
    # 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal
    d["thermal_throttle"] = bool(int(d.get("throttle_reasons", "0x0"), 16) & (0x8 | 0x20 | 0x40))
    return d
