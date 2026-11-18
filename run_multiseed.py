"""run_multiseed.py - Uji kekokohan eksperimen gamma: 5 seed per konfigurasi, evaluasi 500 hari uji."""
import numpy as np, pandas as pd
from bike_env import load_day_types
from qlearning import *
day_types = load_day_types("data/sampleSubmission.csv")["daytype"].values
rows = []
for g in [0.9, 0.5, 0.1, 0.0]:
    for seed in range(5):
        q, r, l, e, v = train(day_types, alpha=0.1, gamma=g, episodes=20000, seed=seed)
        ev = evaluate(q_policy(q)); conv, _ = convergence_episode(r)
        rows.append(dict(gamma=g, seed=seed, reward_eval=ev["reward"], service=ev["service_level_%"], lost=ev["lost"],
                         overflow=ev["overflow"], moves=ev["moves"], konvergensi=conv))
    print("gamma", g, "selesai", flush=True)
df = pd.DataFrame(rows); df.to_csv("results/multiseed_gamma_raw.csv", index=False)
agg = df.groupby("gamma").agg(['mean', 'std']).drop(columns="seed").round(2)
agg.to_csv("results/multiseed_gamma_summary.csv"); print(agg.to_string())
