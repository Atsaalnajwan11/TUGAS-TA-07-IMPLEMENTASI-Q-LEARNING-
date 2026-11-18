"""qlearning.py - Algoritma Q-Learning (tabular) + evaluasi + kebijakan pembanding."""
import numpy as np
import pandas as pd
from bike_env import BikeShareEnv, N_STATES, N_ACTIONS


def train(day_types, alpha=0.1, gamma=0.9, episodes=20000, eps_start=1.0, eps_min=0.01,
          decay_rate=0.0005, seed=42, log_every=0):
    """day_types: array tipe_hari (0/1) yang di-sample untuk tiap episode (berasal dari CSV)."""
    env = BikeShareEnv(seed)
    rng = np.random.default_rng(seed + 1)
    q = np.zeros((N_STATES, N_ACTIONS))
    visits = np.zeros((N_STATES, N_ACTIONS), dtype=int)
    rewards, lost_ep, eps_hist = [], [], []
    for ep in range(episodes):
        epsilon = eps_min + (eps_start - eps_min) * np.exp(-decay_rate * ep)
        s = env.reset(daytype=rng.choice(day_types))
        total, lost, done = 0.0, 0, False
        while not done:
            if rng.random() < epsilon:                      # eksplorasi
                a = int(rng.integers(N_ACTIONS))
            else:                                           # eksploitasi
                row = q[s]; a = int(rng.choice(np.flatnonzero(row == row.max())))
            s2, r, done, info = env.step(a)
            target = r if done else r + gamma * q[s2].max()  # terminal: tanpa bootstrap
            q[s, a] = (1 - alpha) * q[s, a] + alpha * target  # Persamaan Bellman
            visits[s, a] += 1
            s = s2; total += r; lost += info["lost"]
        rewards.append(total); lost_ep.append(lost); eps_hist.append(epsilon)
        if log_every and (ep + 1) % log_every == 0:
            print(f"ep {ep+1:6d}  eps={epsilon:.3f}  MA100 reward={np.mean(rewards[-100:]):8.2f}", flush=True)
    return q, np.array(rewards), np.array(lost_ep), np.array(eps_hist), visits


def evaluate(policy, n_days=500, seed=123, weekend_ratio=2 / 7):
    """policy(state, env) -> action. Mengembalikan metrik rata-rata per hari."""
    env = BikeShareEnv(seed)
    rng = np.random.default_rng(seed)
    m = dict(reward=[], served=[], lost=[], overflow=[], moves=[])
    for _ in range(n_days):
        s = env.reset(daytype=int(rng.random() < weekend_ratio))
        done, tot = False, dict(reward=0, served=0, lost=0, overflow=0, moves=0)
        while not done:
            a = policy(s, env)
            s, r, done, info = env.step(a)
            tot["reward"] += r; tot["served"] += info["served"]; tot["lost"] += info["lost"]
            tot["overflow"] += info["overflow"]; tot["moves"] += int(info["moved"] > 0)
        for k in m: m[k].append(tot[k])
    out = {k: float(np.mean(v)) for k, v in m.items()}
    out["service_level_%"] = 100 * out["served"] / (out["served"] + out["lost"])
    return out


def q_policy(q):
    return lambda s, env: int(np.argmax(q[s]))

def no_move_policy(s, env):
    return 0

def random_policy_factory(seed=0):
    r = np.random.default_rng(seed)
    return lambda s, env: int(r.integers(N_ACTIONS))

def heuristic_policy(s, env):
    """Aturan sederhana: pindahkan dari stasiun paling penuh ke yang paling kosong bila selisih >= 8."""
    hi, lo = int(np.argmax(env.stock)), int(np.argmin(env.stock))
    if env.stock[hi] - env.stock[lo] >= 8:
        return {(0, 1): 1, (0, 2): 2, (1, 0): 3, (1, 2): 4, (2, 0): 5, (2, 1): 6}[(hi, lo)]
    return 0


def convergence_episode(rewards, window=100, smooth=500, tol_frac=0.10):
    """Episode pertama setelah mana moving average (jendela `smooth`) selalu berada dalam
    +-tol_frac x (plateau - reward awal) dari plateau (rata-rata 1000 episode terakhir)."""
    r = np.asarray(rewards)
    ma = pd.Series(r).rolling(smooth, min_periods=1).mean().values
    plateau = r[-1000:].mean()
    tol = tol_frac * abs(plateau - r[:window].mean())
    bad = np.flatnonzero(np.abs(ma - plateau) > tol)
    return (int(bad[-1]) + 1 if len(bad) else 0), plateau
