"""run_training.py - Langkah 1-4 TA-07: latih agen, kurva pembelajaran, ekstrak & visualisasikan policy."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bike_env import *
from qlearning import *

CSV = "data/sampleSubmission.csv"
cal = load_day_types(CSV)
print(f"Kalender dari CSV: {len(cal)} hari ({(cal.daytype==0).sum()} hari kerja, {(cal.daytype==1).sum()} akhir pekan)")
day_types = cal["daytype"].values

EPISODES = 20000
q, rewards, lost, eps, visits = train(day_types, alpha=0.1, gamma=0.9, episodes=EPISODES,
                                      decay_rate=0.0005, seed=42, log_every=2500)
np.save("results/q_table.npy", q)
pd.DataFrame(q, columns=ACTION_NAMES).to_csv("results/q_table.csv", index_label="state")
pd.DataFrame({"episode": range(EPISODES), "reward": rewards, "lost": lost, "epsilon": eps}).to_csv("results/training_log.csv", index=False)

# ---- Langkah 3: kurva pembelajaran ----
W = 100
ma = pd.Series(rewards).rolling(W, min_periods=1).mean()
conv, plateau = convergence_episode(rewards, W)
print(f"Konvergensi (MA{W} stabil +-5%) pada episode ~{conv}, plateau reward = {plateau:.1f}")
print(f"Reward rata-rata 100 episode pertama = {rewards[:100].mean():.1f}; 1000 episode terakhir = {rewards[-1000:].mean():.1f}")
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax[0].plot(rewards, color="lightgray", lw=.4, label="Reward per episode")
ax[0].plot(ma, color="green", lw=1.8, label=f"Moving average ({W})")
ax[0].axvline(conv, color="red", ls="--", label=f"Konvergensi ~ ep {conv}")
ax[0].set_ylabel("Total reward / hari"); ax[0].set_title("Kurva Pembelajaran Q-Learning - Redistribusi Sepeda"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(eps, color="purple"); ax[1].set_ylabel("Epsilon"); ax[1].set_xlabel("Episode"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/learning_curve.png", dpi=150); plt.close()

# ---- Langkah 4: policy ----
policy_idx = q.argmax(1)
# frekuensi kunjungan state di bawah kebijakan greedy (1000 hari uji) -> state langka ditandai abu-abu
greedy_visits = np.zeros(N_STATES, dtype=int)
act_hour = {0: np.zeros((24, N_ACTIONS)), 1: np.zeros((24, N_ACTIONS))}
_env = BikeShareEnv(999); _rng = np.random.default_rng(999)
for _ in range(1000):
    dt = int(_rng.random() < 2/7); st = _env.reset(daytype=dt); dn = False; h = 0
    while not dn:
        a = int(policy_idx[st]); greedy_visits[st] += 1; act_hour[dt][h, a] += 1
        st, _, dn, _ = _env.step(a); h += 1
seen = greedy_visits >= 10
print("State yang sering dikunjungi policy greedy (>=10 kali):", int(seen.sum()), "dari", N_STATES)
LV = ["L", "M", "H"]
def combo(k): return f"{LV[k//9]}{LV[(k//3)%3]}{LV[k%3]}"
fig, axes = plt.subplots(2, 1, figsize=(17, 8))
cmap = plt.get_cmap("tab10", N_ACTIONS).copy(); cmap.set_bad('#dddddd')
for d, name in enumerate(["HARI KERJA", "AKHIR PEKAN"]):
    grid = np.full((N_PERIOD, 27), np.nan); txt = [[""]*27 for _ in range(N_PERIOD)]
    for p in range(N_PERIOD):
        for k in range(27):
            s = ((d * N_PERIOD + p) * 27) + k
            if seen[s]:
                grid[p, k] = policy_idx[s]; txt[p][k] = ACTION_NAMES[policy_idx[s]].replace("Diam", "-")
    ax = axes[d]; ax.imshow(grid, cmap=cmap, vmin=0, vmax=N_ACTIONS-1, aspect="auto")
    for p in range(N_PERIOD):
        for k in range(27): ax.text(k, p, txt[p][k], ha="center", va="center", fontsize=7, color="white")
    ax.set_xticks(range(27)); ax.set_xticklabels([combo(k) for k in range(27)], rotation=90, fontsize=8)
    ax.set_yticks(range(N_PERIOD)); ax.set_yticklabels(PERIOD_NAMES, fontsize=8)
    ax.set_title(f"Kebijakan optimal - {name}  (kolom = level stok [R,K,C]; abu-abu = state jarang dikunjungi)")
plt.tight_layout(); plt.savefig("results/policy_heatmap.png", dpi=150); plt.close()

rows = []
for d in range(2):
    for p in range(N_PERIOD):
        for k in range(27):
            s = ((d * N_PERIOD + p) * 27) + k
            if seen[s]: rows.append(["Hari kerja" if d == 0 else "Akhir pekan", PERIOD_NAMES[p], combo(k), ACTION_NAMES[policy_idx[s]]])
pd.DataFrame(rows, columns=["tipe_hari", "periode", "level_stok_RKC", "aksi"]).to_csv("results/policy_table.csv", index=False)

# ---- Evaluasi vs baseline (500 hari uji, seed berbeda) ----
res = {"Q-Learning": evaluate(q_policy(q)), "Tanpa redistribusi": evaluate(no_move_policy),
       "Heuristik (selisih>=8)": evaluate(heuristic_policy), "Acak": evaluate(random_policy_factory(7))}
df = pd.DataFrame(res).T.round(2); print(df); df.to_csv("results/evaluation.csv")

# ---- Frekuensi aksi per jam (policy greedy) ----
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for d, name in enumerate(["Hari kerja", "Akhir pekan"]):
    frac = act_hour[d] / act_hour[d].sum(1, keepdims=True); bottom = np.zeros(24)
    for a in range(N_ACTIONS):
        ax[d].bar(range(24), frac[:, a], bottom=bottom, color=plt.get_cmap("tab10")(a), label=ACTION_NAMES[a]); bottom += frac[:, a]
    ax[d].set_title(f"Komposisi aksi per jam - {name}"); ax[d].set_xlabel("Jam"); ax[d].set_xticks(range(0, 24, 2))
ax[0].set_ylabel("Proporsi aksi"); ax[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.savefig("results/action_by_hour.png", dpi=150); plt.close()
_ah = pd.DataFrame(act_hour[0] / act_hour[0].sum(1, keepdims=True), columns=ACTION_NAMES).round(2); _ah.to_csv("results/action_by_hour_weekday.csv", index_label="jam")
print(_ah.to_string())

# ---- Dinamika stok satu hari kerja ----
def trace(policy, seed=5):
    env = BikeShareEnv(seed); s = env.reset(daytype=0, init_stock=[10, 10, 10]); H = [env.stock.copy()]; done = False; acts = []
    while not done:
        a = policy(s, env); acts.append(a); s, r, done, _ = env.step(a); H.append(env.stock.copy())
    return np.array(H), acts
fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for a, (t, pol) in zip(ax, [("Tanpa redistribusi", no_move_policy), ("Q-Learning", q_policy(q))]):
    H, acts = trace(pol)
    for i, n in enumerate(STATIONS): a.plot(H[:, i], label=n, marker="o", ms=3)
    a.set_title(f"Stok sepeda hari kerja - {t}"); a.set_xlabel("Jam"); a.grid(alpha=.3); a.axhline(CAP, color="k", ls=":", lw=.8)
    if t == "Q-Learning": print("Aksi per jam (hari kerja):", [ACTION_NAMES[x] for x in acts])
ax[0].set_ylabel("Sepeda di stasiun"); ax[0].legend(); plt.tight_layout(); plt.savefig("results/stock_dynamics.png", dpi=150)
print("Selesai.")
