"""run_experiment.py - Langkah 5 TA-07: eksperimen hiperparameter.
Utama: gamma 0.9 (default/tinggi) vs 0.1 (sangat rendah -> agen miopik).
Bonus: alpha 0.1 (default) vs 0.001 (sangat rendah) dan alpha 0.9 (sangat tinggi)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bike_env import load_day_types
from qlearning import *

day_types = load_day_types("data/sampleSubmission.csv")["daytype"].values
EP, W = 20000, 100
def run(**kw):
    q, r, l, e, v = train(day_types, episodes=EP, seed=42, **kw)
    ev = evaluate(q_policy(q)); conv, plat = convergence_episode(r, W)
    return q, r, ev, conv, plat

def compare(configs, title, fname):
    plt.figure(figsize=(10, 5.5)); rows = []
    for label, kw, col in configs:
        q, r, ev, conv, plat = run(**kw)
        plt.plot(pd.Series(r).rolling(W, min_periods=1).mean(), color=col, label=label)
        rows.append(dict(konfigurasi=label, konvergensi_episode=conv, reward_plateau_train=round(plat, 1),
                         reward_eval=round(ev["reward"], 1), service_level_pct=round(ev["service_level_%"], 1),
                         lost_per_hari=round(ev["lost"], 1), overflow_per_hari=round(ev["overflow"], 1), langkah_truk=round(ev["moves"], 1)))
        print(rows[-1], flush=True)
    plt.title(title); plt.xlabel("Episode"); plt.ylabel(f"Moving average reward ({W} episode)"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"results/{fname}.png", dpi=150); plt.close()
    pd.DataFrame(rows).to_csv(f"results/{fname}.csv", index=False)

compare([("gamma = 0.9 (default)", dict(alpha=0.1, gamma=0.9), "green"),
         ("gamma = 0.1 (sangat rendah)", dict(alpha=0.1, gamma=0.1), "red")],
        "Eksperimen Discount Factor (gamma)", "exp_gamma")
compare([("alpha = 0.1 (default)", dict(alpha=0.1, gamma=0.9), "green"),
         ("alpha = 0.001 (sangat rendah)", dict(alpha=0.001, gamma=0.9), "red"),
         ("alpha = 0.9 (sangat tinggi)", dict(alpha=0.9, gamma=0.9), "orange")],
        "Eksperimen Learning Rate (alpha) - bonus", "exp_alpha")
