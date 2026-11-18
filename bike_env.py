"""
bike_env.py  -  Lingkungan simulasi Redistribusi Sepeda Antarstasiun (Python murni + NumPy)

Skenario   : 3 stasiun bike-sharing (Perumahan, Perkantoran, Kampus), 1 truk redistribusi.
Agen       : Operator/dispatcher truk redistribusi.
Episode    : 1 hari = 24 langkah (1 langkah = 1 jam).
State      : (tipe_hari, periode_waktu, level_stok_R, level_stok_K, level_stok_C)
             2 x 5 x 3 x 3 x 3 = 270 state
Action     : 7 aksi  (0 = diam, 1..6 = pindahkan MOVE sepeda dari stasiun i ke j)
Reward     : +1 per sewa terlayani, -3 per permintaan gagal (stock-out),
             -2 per sepeda balik yang ditolak (dock penuh), -1 biaya per perpindahan truk,
             -1.5 jika aksi tidak valid (truk tidak bisa memindahkan apa pun).
"""
import numpy as np
import pandas as pd

STATIONS = ["Perumahan", "Perkantoran", "Kampus"]
SHORT = ["R", "K", "C"]
N_ST = 3
CAP = 20            # kapasitas dock per stasiun
MOVE = 5            # jumlah sepeda yang dipindahkan truk per aksi
ACTIONS = [None, (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
ACTION_NAMES = ["Diam"] + [f"{SHORT[i]}->{SHORT[j]}" for i, j in ACTIONS[1:]]
N_ACTIONS = len(ACTIONS)
PERIOD_NAMES = ["Malam (00-05)", "Pagi (06-09)", "Siang (10-15)", "Sore (16-19)", "Malam (20-23)"]
N_PERIOD, N_DAYTYPE, N_LEVEL = 5, 2, 3
N_STATES = N_DAYTYPE * N_PERIOD * N_LEVEL ** N_ST     # 270

R_SERVED, R_LOST, R_OVERFLOW, R_MOVE, R_INVALID = 1.0, -3.0, -2.0, -1.0, -1.5


def period_of(hour: int) -> int:
    if hour < 6:  return 0
    if hour < 10: return 1
    if hour < 16: return 2
    if hour < 20: return 3
    return 4


def level_of(stock: int) -> int:
    """Diskritisasi stok: 0=Rendah (<=6), 1=Sedang (7-13), 2=Tinggi (>=14)."""
    return 0 if stock <= 6 else (1 if stock <= 13 else 2)


# --- Parameter permintaan (rata-rata sewa per jam per stasiun), indeks [periode] ---
LAMBDA = {
    0: np.array([[0.3, 0.2, 0.3], [7.0, 1.0, 2.5], [1.5, 3.0, 3.5], [1.5, 7.0, 4.5], [1.5, 0.8, 1.2]]),  # hari kerja
    1: np.array([[0.3, 0.1, 0.2], [1.5, 0.5, 0.8], [3.0, 1.5, 3.0], [3.0, 1.5, 2.5], [1.5, 0.5, 1.0]]),  # akhir pekan
}
_M = lambda a, b, c: np.array([a, b, c], dtype=float)
DEST_WEEKDAY = {
    1: np.array([_M(0, .5, .5), _M(.8, 0, .2), _M(.6, .4, 0)]),     # pagi  : menuju tempat kerja/kampus
    2: np.array([_M(0, .5, .5), _M(.4, 0, .6), _M(.4, .6, 0)]),     # siang : campuran
    3: np.array([_M(0, .5, .5), _M(.85, 0, .15), _M(.8, .2, 0)]),   # sore  : pulang ke perumahan
    0: np.array([_M(0, .5, .5), _M(.9, 0, .1), _M(.9, .1, 0)]),
    4: np.array([_M(0, .5, .5), _M(.9, 0, .1), _M(.9, .1, 0)]),
}
DEST_WEEKEND = np.array([_M(0, .5, .5), _M(.5, 0, .5), _M(.5, .5, 0)])


def load_day_types(csv_path: str):
    """Baca kolom datetime dari CSV -> daftar (tanggal, tipe_hari). 0=hari kerja, 1=akhir pekan.
    Kolom 'count' pada sampleSubmission.csv semuanya 0, jadi TIDAK dipakai sebagai permintaan."""
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    days = df["datetime"].dt.normalize().drop_duplicates().reset_index(drop=True)
    daytype = (days.dt.dayofweek >= 5).astype(int)
    return pd.DataFrame({"date": days, "daytype": daytype})


class BikeShareEnv:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def encode(self):
        p = period_of(min(self.hour, 23))
        l = [level_of(s) for s in self.stock]
        return ((((self.daytype * N_PERIOD + p) * N_LEVEL + l[0]) * N_LEVEL + l[1]) * N_LEVEL + l[2])

    def reset(self, daytype=0, init_stock=None):
        self.daytype, self.hour = int(daytype), 0
        self.stock = (np.array(init_stock) if init_stock is not None
                      else self.rng.integers(5, 16, size=N_ST)).astype(int)
        return self.encode()

    def step(self, action):
        info = dict(served=0, lost=0, overflow=0, moved=0)
        reward = 0.0
        # 1) truk memindahkan sepeda
        if action != 0:
            i, j = ACTIONS[action]
            n = int(min(MOVE, self.stock[i], CAP - self.stock[j]))
            if n > 0:
                self.stock[i] -= n; self.stock[j] += n
                reward += R_MOVE; info["moved"] = n
            else:
                reward += R_INVALID
        # 2) permintaan pelanggan selama 1 jam
        p = period_of(self.hour)
        lam = LAMBDA[self.daytype][p]
        dest = DEST_WEEKEND if self.daytype == 1 else DEST_WEEKDAY[p]
        demand = self.rng.poisson(lam)
        arrivals = np.zeros(N_ST, dtype=int)
        for s in range(N_ST):
            served = int(min(demand[s], self.stock[s]))
            lost = int(demand[s] - served)
            self.stock[s] -= served
            if served:
                arrivals += self.rng.multinomial(served, dest[s])
            info["served"] += served; info["lost"] += lost
        # 3) sepeda dikembalikan ke stasiun tujuan
        for s in range(N_ST):
            ok = int(min(arrivals[s], CAP - self.stock[s]))
            self.stock[s] += ok
            info["overflow"] += int(arrivals[s] - ok)
        reward += R_SERVED * info["served"] + R_LOST * info["lost"] + R_OVERFLOW * info["overflow"]
        self.hour += 1
        done = self.hour >= 24
        return self.encode(), reward, done, info
