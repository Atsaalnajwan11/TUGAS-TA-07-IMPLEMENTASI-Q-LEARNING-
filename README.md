# TA-07 – Redistribusi Sepeda Antarstasiun Bike Sharing (Q-Learning)

Pembelajaran Mesin · Modul 07 Reinforcement Learning

## Struktur proyek
| Berkas | Isi |
|---|---|
| `bike_env.py` | Lingkungan simulasi (S, A, R) – Python murni + NumPy |
| `qlearning.py` | Q-Learning (ε-greedy), evaluasi, kebijakan pembanding, deteksi konvergensi |
| `run_training.py` | Langkah 1–4: pelatihan, kurva pembelajaran, policy, evaluasi |
| `run_experiment.py` | Langkah 5: eksperimen γ (utama) dan α (bonus) |
| `run_multiseed.py` | Uji kekokohan γ dengan 5 seed |
| `TA07_BikeSharing_QLearning.ipynb` | Notebook (sudah dieksekusi) untuk Jupyter/Colab |
| `Laporan_TA07_BikeSharing_QLearning.docx/.pdf` | Laporan |
| `data/sampleSubmission.csv` | Kalender simulasi (kolom `count` semuanya 0, tidak dipakai) |
| `results/` | Grafik PNG, Q-Table, log pelatihan, tabel hasil |

## Cara menjalankan
```bash
pip install numpy pandas matplotlib
python run_training.py      # ± 15 detik
python run_experiment.py    # ± 1 menit
python run_multiseed.py     # ± 3 menit (opsional)
```

## Langkah pengerjaan TA-07
1. **Rancang skenario (lanjutan TP-07).** Tentukan agen (operator truk), lingkungan (3 stasiun), S (270 state), A (7 aksi), R (sewa +1, gagal −3, overflow −2, biaya truk −1). Hitung skalabilitas: 2×5×3×3×3 = 270.
2. **Siapkan data.** Baca kolom `datetime` dari CSV → tentukan tipe hari tiap episode. Karena `count` = 0, permintaan dibangkitkan simulator Poisson (jika punya `train.csv` Kaggle Bike Sharing Demand, parameter `LAMBDA` bisa diestimasi dari sana).
3. **Bangun lingkungan** `BikeShareEnv.step(action)` → `new_state, reward, done, info`.
4. **Implementasi Q-Learning**: Q-Table nol (270×7), ε-greedy dengan decay eksponensial, update Bellman (terminal tanpa bootstrap).
5. **Rekam reward per episode** dan plot moving average (window 100); tentukan episode konvergensi; jelaskan fluktuasi awal.
6. **Ekstrak policy** (`argmax Q`), visualisasikan (heatmap + komposisi aksi per jam + dinamika stok), tafsirkan, dan bandingkan dengan baseline.
7. **Eksperimen hiperparameter**: satu parameter (γ ATAU α), nilai default vs sangat rendah; bandingkan kurva dan metrik. Ulangi dengan beberapa seed bila selisih kecil.
8. **Tulis laporan** (DOC + PDF), unggah kode ke GitHub, rekam video 5–7 menit (opsional).

## Hasil ringkas (seed 42)
Konvergensi ≈ episode 5.143 · Service level 97,2% (tanpa redistribusi 78,5%) · Reward evaluasi 107,9/hari.
Eksperimen γ: γ rendah **tidak** merugikan pada lingkungan ini (lihat laporan bagian 5).

## Catatan
Kode contoh di modul (Langkah 3) memakai `max_epsilon` dan `rewards_per_episode` yang belum didefinisikan; keduanya sudah ditangani di `qlearning.py`.
