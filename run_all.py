"""Full pipeline: train CNN + SNN (T in 5/10/25) x 3 seeds, count ops, energy estimate, latency, FGSM.

    python run_all.py --quick     # ~1 min smoke test -> results/quick/
    python run_all.py             # full run -> results/
"""
import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# MLflow 3.x refuses the file store unless opted in; the brief asks for a local file store (./mlruns).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import snntorch
import torch

import snn_bench as sb

SEEDS, TS, EPS = [0, 1, 2], [5, 10, 25], [0.0, 0.05, 0.1, 0.2]
FGSM_T, LAT_BS, TRAIN_BS, LR = 10, 256, 128, 1e-3
BLUE, ORANGE = "#2a78d6", "#eb6834"  # CNN, SNN
ENERGY_NOTE = "Theoretical estimate for neuromorphic hardware (Horowitz 2014, 45 nm). Not measured; a GPU saves no energy running an SNN."


def env_info():
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"gpu_driver": smi, "python": platform.python_version(), "os": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "snntorch": snntorch.__version__, "mlflow": mlflow.__version__}


def finished_runs(epochs, quick):
    """CSV rows rebuilt from FINISHED MLflow runs (latest per model/T/seed). MLflow is the single source of truth,
    so a killed pipeline resumes by skipping what already finished."""
    df = mlflow.search_runs(filter_string=f"attributes.status = 'FINISHED' and params.epochs = '{epochs}' "
                                          f"and params.quick = '{quick}'", order_by=["attributes.start_time DESC"])
    rows, seen = [], set()
    for _, r in df.iterrows():
        key = (r["params.model"], int(r["params.T"]), int(r["params.seed"]))
        if key in seen:
            continue
        seen.add(key)
        g0, g1 = json.loads(r["params.gpu_before"]), json.loads(r["params.gpu_after"])
        cnn = key[0] == "cnn"
        base = dict(model=key[0], T=key[1], seed=key[2], epochs=epochs,
                    ops_per_inference=r["metrics.ops_per_inference"], op_type="MAC" if cnn else "SOP(AC)",
                    spikes_per_neuron=None if cnn else r["metrics.spikes_per_neuron"],
                    energy_uJ_theoretical=r["metrics.energy_uJ_theoretical"],
                    latency_ms_per_batch=r["metrics.latency_ms_per_batch"], latency_std_ms=r["metrics.latency_std_ms"],
                    latency_batch=int(r["params.latency_bs"]), train_time_s=r["metrics.train_time_s"],
                    gpu_temp_before=g0["temp_c"], gpu_temp_after=g1["temp_c"],
                    sm_clock_before=g0["sm_clock_mhz"], sm_clock_after=g1["sm_clock_mhz"],
                    throttle_reasons_before=g0["throttle_reasons"], throttle_reasons_after=g1["throttle_reasons"],
                    thermal_throttle=g0["thermal_throttle"] or g1["thermal_throttle"],
                    mlflow_run_id=r["run_id"])
        rows.append({**base, "epsilon": 0.0, "accuracy": r["metrics.test_acc"]})
        for e in EPS[1:]:
            c = f"metrics.fgsm_acc_eps_{e}"
            if c in r and pd.notna(r[c]):
                rows.append({**base, "epsilon": e, "accuracy": r[c]})
    return rows, seen


def ms(s):
    return f"{s['mean']:.4g} ± {s['std']:.2g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny subset, 1 epoch: smoke test")
    ap.add_argument("--epochs", type=int, default=10)  # same for all models; full run ~1 h on RTX 4050 Laptop
    a = ap.parse_args()
    t_start = time.time()

    epochs = 1 if a.quick else a.epochs
    warmup, iters = (5, 20) if a.quick else (20, 100)
    out = Path("results/quick" if a.quick else "results")
    out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())  # local file store
    mlflow.set_experiment("snn-efficiency-benchmark" + ("-quick" if a.quick else ""))

    env = env_info()
    (out / "env.json").write_text(json.dumps(env, indent=2))
    macs = sb.self_check()
    print(f"✅ self-check passed; CNN MACs/inference (thop == analytic) = {macs:,}")

    xtr, ytr, xte, yte = sb.load_mnist()
    if a.quick:
        xtr, ytr, xte, yte = xtr[:2000], ytr[:2000], xte[:1000], yte[:1000]
    print(f"✅ MNIST loaded: train {len(xtr)}, test {len(xte)}, epochs {epochs}")

    _, done = finished_runs(epochs, a.quick)
    done = set() if a.quick else done  # quick always reruns; full run resumes
    for key in sorted(done):
        print(f"✅ resumed from MLflow (already finished): {key[0]}_T{key[1]}_seed{key[2]}")
    for seed in SEEDS:
        for name, T in [("cnn", 0)] + [("snn", t) for t in TS]:
            tag = f"{name}_T{T}_seed{seed}"
            if (name, T, seed) in done:
                continue
            with mlflow.start_run(run_name=tag):
                sb.seed_all(seed)
                model = (sb.CNN() if name == "cnn" else sb.SNN(T)).to(sb.DEV)
                t0 = time.time()
                last_loss = sb.train(model, xtr, ytr, epochs, TRAIN_BS, LR)
                train_s = time.time() - t0
                sb.seed_all(seed)  # fixed spike-encoding noise for evaluation
                acc, sops, spk = sb.evaluate(model, xte, yte)
                g0 = sb.gpu_state()
                lat, lat_sd = sb.latency_ms(model, xte[:LAT_BS], warmup, iters)
                g1 = sb.gpu_state()
                ops = macs if name == "cnn" else sops
                energy_uj = ops * (sb.E_MAC_PJ if name == "cnn" else sb.E_AC_PJ) * 1e-6
                fg = {}
                if name == "cnn" or T == FGSM_T:
                    for e in EPS[1:]:
                        sb.seed_all(seed)
                        fg[e] = sb.fgsm_acc(model, xte, yte, e)

                mlflow.log_params(dict(model=name, T=T, seed=seed, epochs=epochs, train_bs=TRAIN_BS, lr=LR,
                                       beta=model.lif[0].beta.item() if name == "snn" else None, surrogate="fast_sigmoid(25)" if name == "snn" else None,
                                       latency_bs=LAT_BS, warmup=warmup, timed_iters=iters,
                                       gpu_before=json.dumps(g0), gpu_after=json.dumps(g1), quick=a.quick))
                mlflow.log_metrics({k: v for k, v in dict(
                    test_acc=acc, ops_per_inference=ops, spikes_per_neuron=spk, energy_uJ_theoretical=energy_uj,
                    latency_ms_per_batch=lat, latency_std_ms=lat_sd, train_time_s=train_s, final_train_loss=last_loss,
                    **{f"fgsm_acc_eps_{e}": v for e, v in fg.items()}).items() if v is not None})
            spk_s = f", spikes/neuron {spk:.3f}" if spk is not None else ""
            fg_s = ", FGSM " + " ".join(f"{e}:{v:.4f}" for e, v in fg.items()) if fg else ""
            print(f"✅ {tag}: acc {acc:.4f}, ops {ops:,.0f}{spk_s}, E_est {energy_uj:.3f} uJ, "
                  f"lat {lat:.3f} ms/batch, train {train_s:.0f}s, temp {g0['temp_c']}->{g1['temp_c']}C, "
                  f"throttle {g0['thermal_throttle'] or g1['thermal_throttle']}{fg_s}", flush=True)

    rows, seen = finished_runs(epochs, a.quick)
    assert len(seen) == len(SEEDS) * (1 + len(TS)), f"missing runs: only {sorted(seen)} finished"
    df = pd.DataFrame(rows).sort_values(["model", "T", "seed", "epsilon"])
    df.to_csv(out / "results.csv", index=False)

    # ---- aggregate (mean ± sample std over seeds, n=3) ----
    agg = lambda col, d: d.groupby(["model", "T"])[col].agg(["mean", "std"])
    clean = df[df.epsilon == 0]
    tab = pd.concat({c: agg(c, clean) for c in ["accuracy", "spikes_per_neuron", "ops_per_inference",
                                                 "energy_uJ_theoretical", "latency_ms_per_batch"]}, axis=1)
    fg_tab = df[(df.model == "cnn") | (df["T"] == FGSM_T)].groupby(["model", "T", "epsilon"])["accuracy"].agg(["mean", "std"])

    cnn_e = tab.loc[("cnn", 0), ("energy_uJ_theoretical", "mean")]
    lines = ["| Model | T | Test acc | Spikes/neuron/inference | Ops/inference | Energy est. (µJ, theoretical) | CNN/SNN energy ratio | GPU latency (ms / batch 256) |",
             "|---|---|---|---|---|---|---|---|"]
    for (m, T), r in tab.iterrows():
        g = lambda c: {"mean": r[(c, "mean")], "std": r[(c, "std")]}
        spk = "–" if m == "cnn" else ms(g("spikes_per_neuron"))
        opn = ("MACs " if m == "cnn" else "SOPs ") + ms(g("ops_per_inference"))
        ratio = "1.00" if m == "cnn" else f"{cnn_e / r[('energy_uJ_theoretical', 'mean')]:.2f}×"
        lines.append(f"| {m.upper()} | {T or '–'} | {ms(g('accuracy'))} | {spk} | {opn} | {ms(g('energy_uJ_theoretical'))} | {ratio} | {ms(g('latency_ms_per_batch'))} |")
    lines += ["", f"FGSM accuracy (SNN at T={FGSM_T}), mean ± std over seeds:", "",
              "| Model | " + " | ".join(f"ε={e}" for e in EPS) + " |", "|---|" + "---|" * len(EPS)]
    for (m, T), sub in fg_tab.groupby(level=[0, 1]):
        lines.append(f"| {m.upper()}{'' if m == 'cnn' else f' T={T}'} | " +
                     " | ".join(ms(sub.loc[(m, T, e)]) for e in EPS) + " |")
    runs = df.drop_duplicates(["model", "T", "seed"])
    total_min = (time.time() - t_start) / 60
    lines += ["", f"Energy: {ENERGY_NOTE}",
              f"Epochs per run: {epochs}. Runs with thermal throttle flag during latency: {int(runs.thermal_throttle.sum())}/{len(runs)}.",
              f"Runtime of this invocation: {total_min:.1f} min (runs resumed from MLflow: {len(done)}). "
              f"Sum of training time over all {len(runs)} runs: {runs.train_time_s.sum() / 60:.1f} min. "
              f"GPU/driver: {env['gpu_driver']}"]
    summary = "\n".join(lines)
    (out / "summary.md").write_text(summary, encoding="utf-8")

    # ---- plots ----
    style = dict(marker="o", ms=8, lw=2, capsize=4)

    def save(fig, ax, name):
        ax.legend(frameon=False); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(out / name, dpi=150)

    snn_t, cnn_r = tab.loc["snn"], tab.loc[("cnn", 0)]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.errorbar(TS, snn_t[("accuracy", "mean")] * 100, yerr=snn_t[("accuracy", "std")] * 100, color=ORANGE,
                label="SNN (mean ± std, 3 seeds)", **style)
    ax.axhline(cnn_r[("accuracy", "mean")] * 100, color=BLUE, lw=2, ls="--", label="CNN (mean, 3 seeds)")
    ax.set(xlabel="Time steps T", ylabel="Test accuracy (%)", xticks=TS, title="MNIST test accuracy vs T")
    save(fig, ax, "accuracy_vs_T.png")

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.errorbar(TS, snn_t[("ops_per_inference", "mean")], yerr=snn_t[("ops_per_inference", "std")], color=ORANGE,
                label="SNN SOPs (accumulates)", **style)
    ax.axhline(macs, color=BLUE, lw=2, ls="--", label="CNN MACs")
    ax.set(xlabel="Time steps T", ylabel="Operations per inference", xticks=TS, title="SNN SOPs vs CNN MACs")
    save(fig, ax, "sops_vs_T.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["CNN"] + [f"SNN T={t}" for t in TS]
    vals = [cnn_r[("energy_uJ_theoretical", "mean")]] + list(snn_t[("energy_uJ_theoretical", "mean")])
    errs = [cnn_r[("energy_uJ_theoretical", "std")]] + list(snn_t[("energy_uJ_theoretical", "std")])
    bars = ax.bar(labels, vals, yerr=errs, capsize=4, color=[BLUE] + [ORANGE] * 3, width=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(ylabel="Energy per inference (µJ)", title="THEORETICAL energy estimate (not measured)")
    fig.text(0.5, 0.01, "Horowitz 2014, 45 nm: 4.6 pJ/MAC, 0.9 pJ/AC. Neuromorphic-HW estimate; GPU saves no energy.",
             ha="center", fontsize=7.5)
    ax.grid(alpha=0.25, axis="y"); fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out / "energy_estimate.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for (m, T), sub in fg_tab.groupby(level=[0, 1]):
        ax.errorbar(EPS, sub["mean"].values * 100, yerr=sub["std"].values * 100,
                    color=BLUE if m == "cnn" else ORANGE, label="CNN" if m == "cnn" else f"SNN T={T}", **style)
    ax.set(xlabel="FGSM ε (pixel scale [0,1])", ylabel="Test accuracy (%)", xticks=EPS,
           title="FGSM robustness (gradient masking caveat)")
    save(fig, ax, "accuracy_vs_fgsm_eps.png")
    plt.close("all")

    print("\n================ SUMMARY (paste back) ================")
    print(summary)


if __name__ == "__main__":
    main()
