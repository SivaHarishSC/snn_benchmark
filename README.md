# Spiking vs normal neural network: an honest efficiency and robustness benchmark

**In one sentence:** I trained a normal neural network (CNN) and a brain-inspired *spiking* neural network (SNN) on the same task. I compared how accurate they are, how much work they do, how much energy they *would* use on special chips, how fast they run on my laptop GPU, and how they hold up against a simple attack.

Every number in this file was produced by the code in this folder. Nothing is rounded up, guessed or "expected".

---

## 1. The idea in plain words

**The task: MNIST.** The models look at a 28×28 picture of a handwritten digit and say which digit (0–9) it is. MNIST is the classic "hello world" dataset of machine learning: 60,000 training pictures and 10,000 test pictures.

**CNN (Convolutional Neural Network).** The normal kind of image network.
- Every neuron outputs a number, every time.
- Each connection does one **MAC** (multiply-accumulate: `output += weight × input`).
- MACs are expensive in energy.

**SNN (Spiking Neural Network).** Works more like real brain cells.
- A neuron collects input over several **time steps (T)**. It only "fires" a **spike** (a 1) when its internal charge (the *membrane potential*) passes a threshold. Otherwise it stays silent (a 0).
- The neuron model is called **LIF**: *Leaky Integrate-and-Fire*. Its charge slowly leaks away, controlled by a number called **β (beta)**.
- A spike is just a 1, so a connection only needs to *add* its weight, with no multiplication. This is called a synaptic operation (**SOP**), and it is cheaper.
- Silent neurons cost nothing.
- That is why SNNs are interesting for battery-powered devices: drones, robots, sensors.

**The catch.** These savings only happen on special **neuromorphic chips** (e.g. Intel Loihi). A normal GPU computes everything densely, every time step, so on a GPU an SNN is actually *slower*. This project measures both sides honestly.

**How a picture becomes spikes ("rate coding").** At each time step, each pixel fires a spike with probability equal to its brightness. A white pixel spikes almost every step, a black pixel never does.

**How you train an SNN ("surrogate gradient").** Neural networks learn with gradients (slopes). A spike is a hard on/off step, which has no useful slope. The trick is to use a smooth "fake" slope (here `fast_sigmoid`) during learning only. This lets the SNN be trained directly, instead of copying a trained CNN.

**The security part ("FGSM attack").** FGSM, the *Fast Gradient Sign Method*, is the classic adversarial attack.
- It nudges every pixel a tiny amount (**ε, epsilon**) in exactly the direction that confuses the model most.
- To a human the picture looks the same, but the model can fail.
- We test how accuracy drops as ε grows.

---

## 2. What I built

Both models have the **same layer structure**, so the comparison is fair:

```
picture 28×28 → Conv 5×5 (12 filters) → MaxPool → Conv 5×5 (64 filters) → MaxPool → Dense 1024→128 → Dense 128→10 → digit
```

| | CNN | SNN |
|---|---|---|
| Activation | ReLU | LIF spiking neuron (snnTorch `Leaky`, β = 0.9, threshold 1) |
| Input | the picture | the picture turned into random spikes over T steps |
| Output | 10 scores | how many times each of the 10 output neurons spiked; the most spikes wins |
| Training | normal backprop | backprop with surrogate gradient `fast_sigmoid(slope=25)` |

**Settings, identical for both models where possible:**
- **Optimiser:** Adam, learning rate 0.001, batch 128.
- **Epochs:** **10 epochs** for every model. One epoch means seeing all 60,000 pictures once.
  - Why 10: one epoch took 2.5 s for the CNN and 16 / 30 / 84 s for the SNN at T = 5 / 10 / 25. 10 epochs × 3 seeds keeps the whole pipeline around 1 hour, inside my 3-hour budget.
- **Time steps tested:** T = 5, 10 and 25. More steps give the SNN more time to "think", but cost more work.
- **Seeds:** every configuration ran with **3 random seeds** (0, 1, 2). Results are shown as **mean ± standard deviation**, so you can see how much is luck.

---

## 3. What was measured, and how

| Measurement | What it means | How it was done |
|---|---|---|
| **Accuracy** | % of the 10,000 test digits classified correctly | standard test set |
| **MACs** (CNN) | how many multiply-adds one prediction needs | counted with `thop`, and checked against a hand calculation: **1,533,952** |
| **SOPs** (SNN) | how many "spike arrives → add a weight" events one prediction needs | counted exactly: every spike × every connection it travels along, over all T steps |
| **Spikes per neuron** | how active the SNN is (lower = sparser = cheaper) | total spikes ÷ 2,890 neurons, per picture |
| **Energy estimate** | how much energy one prediction *would* use on a chip | MACs × 4.6 pJ, or SOPs × 0.9 pJ (numbers from Horowitz, ISSCC 2014, 45 nm chips). **A theoretical estimate, not a measurement.** |
| **GPU latency** | how long my laptop GPU really takes | batch of 256 pictures; 20 warm-up runs first, then 100 timed runs. `nvidia-smi` checked before and after for overheating (thermal throttling). |
| **FGSM accuracy** | accuracy when pictures are attacked with strength ε = 0, 0.05, 0.1, 0.2 | CNN and SNN (T = 10), all 10,000 test pictures |

**Tracking:** every one of the 12 training runs (CNN + 3 SNNs, × 3 seeds) is logged to **MLflow**, an experiment-tracking tool, in the local folder `mlruns/`.

**Built-in check:** the code verifies its own counters on every run. The SOP counter, fed an "all pixels on" input, must give exactly the dense MAC count, and `thop` must agree with the hand calculation.

---

## 4. Results

### Accuracy, work and energy (mean ± std over 3 seeds)

| Model | T | Test accuracy | Spikes / neuron | Work per prediction | Energy estimate (µJ, **theoretical**) | Energy saving vs CNN | GPU time (ms per 256 pictures) |
|---|---|---|---|---|---|---|---|
| CNN | – | **99.15%** ± 0.04 | – | 1.534 M MACs | 7.056 | 1.00× | 0.85 ± 0.37 |
| SNN | 5 | **98.41%** ± 0.30 | 0.645 ± 0.033 | 1.347 M ± 0.11 M SOPs | 1.212 ± 0.097 | 5.82× | 14.3 ± 4.1 |
| SNN | 10 | **98.88%** ± 0.17 | 1.021 ± 0.12 | 2.169 M ± 0.31 M SOPs | 1.952 ± 0.28 | 3.61× | 33.0 ± 2.8 |
| SNN | 25 | **98.92%** ± 0.23 | 1.828 ± 0.15 | 4.282 M ± 0.45 M SOPs | 3.854 ± 0.40 | 1.83× | 73.5 ± 33 |

### Accuracy under the FGSM attack (SNN with T = 10)

| Model | ε = 0 (no attack) | ε = 0.05 | ε = 0.1 | ε = 0.2 |
|---|---|---|---|---|
| CNN | 99.15% ± 0.04 | 95.34% ± 0.59 | 81.76% ± 3.2 | 36.14% ± 4.3 |
| SNN T=10 | 98.88% ± 0.17 | 97.05% ± 0.37 | 92.82% ± 0.39 | 73.94% ± 4.8 |

**Raw numbers:** `results/results.csv` (one row per model, T, seed and ε) and `results/summary.md`.

### Plots (in `results/`)

| Plot | Shows |
|---|---|
| `accuracy_vs_T.png` | SNN accuracy for T = 5, 10, 25 vs the CNN line |
| `sops_vs_T.png` | SNN work (SOPs) growing with T vs CNN work (MACs) |
| `energy_estimate.png` | theoretical energy per prediction, CNN vs SNN |
| `accuracy_vs_fgsm_eps.png` | how accuracy falls as the attack gets stronger |

---

## 5. What I learned (findings)

1. **The SNN is almost as accurate, and more time steps stop helping after T = 10.**
   - The SNN scores 98.41% (T = 5) up to 98.92% (T = 25), against the CNN's 99.15%.
   - Going from T = 10 to T = 25 gained only 0.04 percentage points, smaller than the seed-to-seed noise, while doubling the work.

2. **More time steps means more work, and at T = 25 the SNN does *more* operations than the CNN.**
   - SOPs per prediction: 1.35 M → 2.17 M → 4.28 M, against 1.53 M CNN MACs.
   - The SNN still "wins" on estimated energy only because an add (0.9 pJ) is assumed to be about 5× cheaper than a multiply-add (4.6 pJ).
   - The estimated saving shrinks from 5.82× (T = 5) to 1.83× (T = 25).
   - Under this model the SNN would lose at about 7.8 M SOPs. The trend suggests a larger T would get there, but I did not test T > 25.

3. **On a normal GPU the SNN is much slower, not faster.**
   - 0.85 ms (CNN) vs 14.3 / 33.0 / 73.5 ms (SNN at T = 5 / 10 / 25) per batch of 256 pictures. That is about 17× / 39× / 87× slower.
   - The GPU computes every time step fully, so the spike sparsity saves nothing there.
   - Training is slower the same way: 8–16 s for the CNN vs 290–304 s for the SNN at T = 10 (10 epochs).
   - **A GPU does not save energy by running an SNN.**

4. **The SNN looks more robust to FGSM, but that is probably an illusion ("gradient masking").**
   - At ε = 0.2 the SNN keeps 73.94% accuracy and the CNN only 36.14%.
   - But FGSM needs the model's gradient. The SNN's gradient is a fake (surrogate) one, and its spike input is random. That makes the attack *aim badly*, rather than the model being truly strong.
   - This effect is called **gradient masking**. These numbers are **not proof that SNNs are more secure**.
   - A real security claim would need stronger attacks: multi-step PGD, averaging over the randomness (EOT), and transfer attacks crafted on the CNN.

5. **A failure worth reporting: the first SNN setting did not learn at all.**
   - With β = 0.5 (charge leaks away fast) and only T = 5 steps, neurons almost never reached the threshold: 0.016 spikes per neuron at the start.
   - With almost no spikes, there is almost no signal to learn from. Accuracy was 9.8% after one epoch, which is random guessing.
   - Changing to β = 0.9 (charge leaks slowly) gave 0.096 spikes per neuron at the start and 96.57% after one epoch.
   - All results above use β = 0.9 for every T.
   - This check was a one-off: 1 seed, 1 epoch, not in `results.csv`.

### How trustworthy are the GPU timings?

- **No overheating.** No thermal-throttling flag was raised in any run, and GPU temperature stayed at 63–74 °C.
- **But the timings were noisy.** The same SNN at T = 25 took 36.97, 82.83 and 100.65 ms across the 3 seeds.
- **Why:** laptop GPUs change their clock speed on their own to save power.
  - In 6 of 9 SNN runs, `nvidia-smi` read a lower clock *after* timing (2490 → 1560–2070 MHz). The GPU was already slowing down while idle.
  - The two runs that started with the "power cap" flag set were the fastest at their T.
- **So:** treat the GPU timings as "roughly how many times slower", not as exact milliseconds.

---

## 6. What went wrong during the project, and how it was handled

| Problem | What happened | Fix |
|---|---|---|
| SNN did not learn at T = 5 | β = 0.5 kept the network almost silent (9.8% accuracy) | β = 0.9 for all T (see finding 5) |
| MLflow refused to start | MLflow 3.18 blocks the old "file store" format unless you opt in | the script sets `MLFLOW_ALLOW_FILE_STORE=true`, keeping the requested local file store |
| Full run was stopped halfway | After about 41 min (10 of 12 runs done), the AI coding tool running the job stopped it because the PC was low on RAM. It was not a bug in this code. | `run_all.py` now rebuilds results from MLflow and **skips runs that already finished**. The resume trained the last 2 runs in 16.9 min. |

**Why resuming did not change the results:** every run sets its own random seed, so run order does not matter. The resumed runs ran in a separate session, which only affects their GPU timing.

**Total compute:**
- 57.0 min of training across all 12 runs.
- Pipeline wall-clock: about 41 min in the first session (09:54–10:35, from log timestamps) plus 16.9 min in the resume.

Logs: `results/full_run.log` and `results/full_run_resume.log`.

---

## 7. Limitations: what this project does NOT show

- **Only MNIST and one small network.** Real tasks (e.g. event cameras on robots) are much harder.
- **The energy numbers are a paper estimate.** They only count the arithmetic. They ignore:
  - memory access, which often costs far more energy than the arithmetic,
  - neuron state updates,
  - pooling,
  - spike routing.

  So the SNN's energy advantage is most likely *overstated*.
- **No neuromorphic chip was used, and no power was measured.** The GPU is not neuromorphic hardware.
- **Only one, weak attack (FGSM),** and the SNN result is likely affected by gradient masking. This is **not a security proof**.
- **Only 3 seeds and 10 epochs.** Models may not be fully trained. The same optimizer settings were used for both models and were not tuned per model. β was changed once, as described above.
- **GPU timings on a laptop are noisy** (see above), and only batch size 256 was tested.

---

## 8. Run it yourself

**Requirements:**
- Windows (or Linux) with an NVIDIA GPU.
- Python 3.12.
- About 1 hour for the full run.
- Tested on an RTX 4050 Laptop GPU (6 GB), driver 566.07.

```powershell
# 1. Create an isolated Python environment
py -3.12 -m venv .venv

# 2. Install the exact package versions (PyTorch with CUDA 12.6)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# 3. Quick smoke test (~1 minute, tiny data, 1 epoch) -> results/quick/
.\.venv\Scripts\python.exe run_all.py --quick

# 4. Full benchmark (~1 hour) -> results/
.\.venv\Scripts\python.exe run_all.py

# 5. (Optional) browse all runs in the MLflow web UI
$env:MLFLOW_ALLOW_FILE_STORE="true"; .\.venv\Scripts\mlflow.exe ui --backend-store-uri ./mlruns
```

**Notes:**
- MNIST downloads automatically on the first run.
- If a full run is interrupted, just run step 4 again. It continues where it stopped.
- To start completely fresh, delete the `mlruns/` folder.

**Exact versions used:**

| | |
|---|---|
| Python | 3.12.10 |
| torch | 2.14.1+cu126 (CUDA 12.6, cuDNN 9.10.2) |
| torchvision | 0.29.1 |
| snntorch | 1.0.0 |
| mlflow | 3.18.0 |
| thop | 0.1.1 |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| matplotlib | 3.11.2 |
| OS | Windows 11 |

Also recorded in `results/env.json`.

---

## 9. Files

| File | What it does |
|---|---|
| `snn_bench.py` | the building blocks: both models, spike/SOP counting, training, testing, the FGSM attack, GPU timing, `nvidia-smi` checks, self-check |
| `run_all.py` | runs everything: 12 runs, MLflow logging, the results CSV, tables, plots and the printed summary |
| `requirements.txt` | exact package versions |
| `results/` | the final CSV, summary, plots, environment info and run logs |

---

## Glossary

| Term | Meaning |
|---|---|
| **Accumulate (AC)** | just an addition; what an SNN connection does when a spike arrives |
| **β (beta)** | how much charge an LIF neuron keeps each step (0.9 = keeps 90%) |
| **Epoch** | one pass over all training pictures |
| **ε (epsilon)** | how much the attack may change each pixel (0.2 = 20% of full brightness) |
| **Gradient masking** | when a model *looks* robust only because attackers get a bad gradient to aim with |
| **Latency** | how long a prediction takes |
| **MAC** | multiply-accumulate, the basic operation of a normal network |
| **Neuromorphic hardware** | chips designed to run spiking networks, where silence costs no energy |
| **pJ / µJ** | picojoule / microjoule; 1 µJ = 1,000,000 pJ |
| **Seed** | the starting value for randomness; different seeds show how much results vary by luck |
| **SOP** | synaptic operation: one spike travelling along one connection (one addition) |
| **T (time steps)** | how many steps the SNN gets to look at the picture |
| **Thermal throttling** | a GPU slowing itself down because it is too hot |
