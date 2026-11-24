# Autonomous Warehouse Navigation with Deep Q-Learning
## Technical Engineering Report & Empirical Evaluation

---

## 1. Judul Project
**Autonomous Warehouse Navigation with Deep Q-Learning (DQN): Perancangan, Implementasi, dan Evaluasi Komparatif Robotika Gudang Otonom Berbasis Reinforcement Learning**

---

## 2. Ringkasan Eksekutif

### 2.1 Tujuan
Tujuan dari project ini adalah merancang, mengimplementasikan, melatih, dan mengevaluasi agent robot gudang otonom (*Automated Guided Vehicle* / AGV) menggunakan algoritma **Deep Q-Network (DQN)**. Agent dilatih untuk melakukan navigasi bebas tabrakan pada lingkungan gudang berstruktur grid 2D, menghindari rak penyimpanan statis (*storage racks*), dan mencapai titik tujuan (*drop-off bay*) secara efisien melalui proses interaksi murni tanpa *hardcoded heuristics*.

### 2.2 Pendekatan
Pendekatan yang diterapkan memadukan:
1. **Gym-like Environment**: Simulasi gudang grid discrete 2D dengan deteksi tabrakan fisik, kendala batas lingkungan, serta pelacakan lintasan gerak aktual.
2. **Feature Engineering Vektor Observasi (19 Dimensi)**: Menggabungkan koordinat posisi normal, vektor anjakan relatif (*relative displacement vector*), sensor lidar batas lokal $3 \times 3$, dan sensor penjejak jarak (*ray-casting clearance sensors*) 4 arah mata angin.
3. **Deep Q-Network (DQN)**: Jaringan saraf tiruan Multi-Layer Perceptron (MLP) yang distabilkan menggunakan *Target Network* dengan pembaruan *soft Polyak update* ($\tau = 0.005$), *Experience Replay Buffer* (kapasitas 40.000 transisi), fungsi *loss* Huber (Smooth L1), dan mekanisme eksplorasi *Epsilon-Greedy* bertahap.
4. **Reward Shaping Potensial**: Menerapkan fungsi reward berbasis penalti langkah diferensial dan shaping jarak Manhattan yang mempertahankan invarian kebijakan optimal (*policy invariance*).
5. **Evaluasi Komparatif Formal**: Menguji performa DQN secara deterministik terhadap algoritma pencarian jalur klasik optimal, yaitu **Breadth-First Search (BFS)** dan **A\* Search** pada 4 skenario denah gudang berbeda (*Simple, Moderate, Complex,* dan *Generalization*).

### 2.3 Hasil Utama
- **Konvergensi Pembelajaran**: Model DQN berhasil mencapai *Success Rate* 100% pada denah pelatihan (*Moderate Map*) dalam rentang 150 episode pertama dan menstabilkan reward kumulatif rata-rata pada $+64.9$ poin pada episode ke-600.
- **Efisiensi Jalur**: Pada evaluasi deterministik di denah gudang utama (*Moderate Map*), DQN menghasilkan jalur sepanjang **22 langkah dengan 0 tabrakan**, menyamai panjang lintasan teoritis optimal yang dihasilkan oleh BFS dan A\*.
- **Transfer Kemampuan Nol-Sampel (*Zero-Shot Generalization*)**: Pada *Simple Map* (10x10), model mencapai *Success Rate* 100% dengan panjang jalur 18 langkah (identik dengan A\*) dan 0 tabrakan.
- **Latensi Inferensi**: Waktu komputasi aksi per langkah model DQN pada perangkat CPU mencapai rata-rata **0,150 ms**, siap untuk kebutuhan pengendalian *real-time*.
- **Analisis Keterbatasan**: Pada denah *Complex Map* (koridor labirin sempit dengan *cul-de-sac*) dan *Generalization Map* (konfigurasi rak baru), agent mengalami fenomena *local obstacle entrapment* (0% success rate), mengonfirmasi bahwa representasi persepsi lokal tanpa mekanisme memori spasial global rentan terhadap jebakan jalan buntu.

### 2.4 Kesimpulan Singkat
DQN terbukti sangat efektif dalam mempelajari navigasi gudang optimal pada topologi rak teratur dan koridor terbuka. Namun, untuk menangani geometri labirin non-konveks yang ekstrem tanpa pemetaan graf global, integrasi representasi peta global (seperti *occupancy grid map* via CNN) atau pendekatan *hierarchical reinforcement learning* diperlukan.

---

## 3. Latar Belakang dan Rumusan Masalah
Dalam era otomatisasi logistik dan *Industry 4.0*, sistem robotika bergerak otonom seperti AGV (*Automated Guided Vehicles*) dan AMR (*Autonomous Mobile Robots*) merupakan tulang punggung efisiensi operasional gudang modern (*fulfillment centers*). 

Metode perencanaan jalur konvensional seperti Dijkstra, BFS, atau A\* memerlukan informasi peta global yang lengkap (*complete global topology*) dan komputasi ulang graf setiap kali terjadi perubahan lingkungan. Sebaliknya, pendekatan *Reinforcement Learning* (RL) menawarkan paradigma di mana agent belajar secara mandiri melalui *trial-and-error*, memetakan persepsi sensorik lokal dan posisi relatif ke estimasi nilai aksi (*action-value / Q-value*).

**Rumusan Masalah:**
1. Bagaimana merancang representasi *state/observation* berbasis vektor yang informatif sehingga agent mampu memahami jarak target sekaligus mendeteksi rintangan terdekat secara efisien?
2. Bagaimana merancang fungsi reward yang memotivasi agent menemukan lintasan terpendek tanpa memicu perilaku eksploitasi looping (*reward loops*)?
3. Seberapa stabil konvergensi model DQN berbasis PyTorch dengan replay buffer dan target network pada lingkungan grid gudang?
4. Bagaimana perbandingan performa terukur antara DQN dan algoritma jalur terpendek deterministik (BFS & A\*) dalam hal tingkat keberhasilan, jumlah tabrakan, panjang jalur, dan latensi inferensi?
5. Bagaimana kemampuan generalisasi model terhadap tata letak gudang yang belum pernah ditemui sebelumnya?

---

## 4. Tujuan dan Ruang Lingkup

### 4.1 Tujuan Spesifik
1. Membangun modul simulasi grid gudang modular (`environment/warehouse_env.py`) yang merepresentasikan komponen fisik gudang.
2. Mengimplementasikan baseline pencarian graf optimal BFS dan A\* (`algorithms/pathfinding.py`) untuk validasi rute dan pembanding performa.
3. Mengembangkan arsitektur Deep Q-Network (`agents/dqn_agent.py`, `agents/q_network.py`, `agents/replay_buffer.py`) dengan pipeline training dan evaluasi modular.
4. Melakukan pelatihan empiris dan pengujian deterministik pada 4 variasi topologi denah gudang.
5. Menghasilkan visualisasi analitis lengkap (grafik training, overlay lintasan, dan animasi GIF navigasi robot).
6. Menyediakan dashboard interaktif Streamlit (`app/streamlit_app.py`) dengan dark mode dan dokumentasi tangkapan layar penuh (*full-page screenshot*).

### 4.2 Ruang Lingkup
- Lingkungan simulasi berupa matriks grid 2D diskret (10x10 dan 12x12).
- Rintangan dimodelkan sebagai rak gudang statis (*static storage racks*).
- Aksi robot bersifat diskret 4-arah (Atas, Bawah, Kiri, Kanan).
- Model komputasi difokuskan pada efisiensi CPU dan kesesuaian reproduktibilitas tanpa memerlukan GPU berkekuatan tinggi.

---

## 5. Desain Environment

### 5.1 Representasi Grid
Lingkungan gudang dimodelkan sebagai matriks 2D ukuran $H \times W$. Koordinat dinyatakan dalam indeks baris dan kolom $(r, c)$, dengan $(0, 0)$ pada sudut kiri atas:
- `0`: *Free cell* (koridor/lorong yang dapat dilalui robot)
- `1`: *Storage rack* (rak penyimpanan barang yang merupakan rintangan statis)
- `2`: *Start location* (titik awal robot / stasiun pengisian)
- `3`: *Goal bay* (lokasi tujuan bongkar-muat / *drop-off bay*)
- `4`: *Robot position* (lokasi aktif robot saat visualisasi)

Matriks rintangan disimpan secara independen dari status koordinat robot, sehingga transisi gerak tidak merusak integritas representasi denah lingkungan.

### 5.2 Komponen Gudang
Sistem menyediakan 4 skenario denah standar di `environment/maps.py`:
1. **Simple Map (10x10)**: Denah gudang terbuka dengan blok rak penyimpanan terisolasi. Jarak Manhattan teoritis: 18 langkah.
2. **Moderate Map (12x12)**: Denah gudang representatif logistik dengan 3 lorong vertikal panjang (*aisles*) dan lorong potong horizontal (*cross-aisles*) di bagian atas, tengah, dan bawah. Jarak Manhattan teoritis: 22 langkah.
3. **Complex Map (12x12)**: Denah dengan lorong sempit, partisi asimetris, dan jalur berbelok dengan jalan buntu (*cul-de-sacs*). Jarak Manhattan teoritis: 20 langkah.
4. **Generalization Map (12x12)**: Denah dengan tata letak rak alternatif dan posisi tujuan di kuadran berlawanan, dirancang untuk menguji transfer pengetahuan kebijakan tanpa pelatihan ulang.

Sebelum setiap eksperimen dijalankan, modul validasi solvabilitas (`validate_map_solvability`) memverifikasi keberadaan minimal satu jalur bebas tabrakan dari *start* ke *goal* menggunakan BFS.

### 5.3 State / Observation Space (19 Dimensi)
Untuk menghasilkan representasi yang ringkas, terukur, dan tidak bergantung pada ukuran matriks grid yang kaku, dirancang vektor fitur bernilai riil 19 dimensi yang dinormalisasi ke rentang $[0.0, 1.0]$:

| No. | Komponen Fitur | Dimensi | Deskripsi Matematis |
| :---: | :--- | :---: | :--- |
| 1–2 | Posisi Robot Ternormalisasi | 2 | $r / (H - 1), \quad c / (W - 1)$ |
| 3–4 | Posisi Goal Ternormalisasi | 2 | $r_g / (H - 1), \quad c_g / (W - 1)$ |
| 5–6 | Vektor Anjakan Relatif | 2 | $(r_g - r) / (H - 1), \quad (c_g - c) / (W - 1)$ |
| 7 | Jarak Manhattan Ternormalisasi | 1 | $(\|r_g - r\| + \|c_g - c\|) / (H + W - 2)$ |
| 8–15 | Sensor Okupansi Lokal $3 \times 3$ | 8 | Status biner tetangga 8-arah (N, NE, E, SE, S, SW, W, NW); $1.0$ jika rak/dinding, $0.0$ jika bebas. |
| 16–19 | Sensor Ray-Casting 4 Arah | 4 | Jarak bebas langsung hingga membentur rak/dinding pada arah $[UP, DOWN, LEFT, RIGHT]$ dinormalisasi oleh $\max(H, W)$. |

Representasi ini memberikan agent kesadaran ganda: kesadaran lokal (*local collision avoidance*) dan kesadaran global (*global orientation toward target*).

### 5.4 Action Space
Ruang aksi diskret memiliki kardinalitas 4:
- `0`: Bergerak ke Atas ($\Delta r = -1, \Delta c = 0$)
- `1`: Bergerak ke Bawah ($\Delta r = +1, \Delta c = 0$)
- `2`: Bergerak ke Kiri ($\Delta r = 0, \Delta c = -1$)
- `3`: Bergerak ke Kanan ($\Delta r = 0, \Delta c = +1$)

Jika aksi mengarahkan robot ke luar batas grid atau menabrak sel rak gudang:
- Robot **tetap berada di posisi sel semula** (*no invalid displacement*).
- Penghitung tabrakan (*collision counter*) bertambah 1.
- Dikenakan penalti tabrakan.

### 5.5 Reward Function
Fungsi reward dirancang secara hati-hati untuk mendorong pencapaian tujuan melalui jalur terpendek tanpa memicu exploitasi looping:

$$R(s, a, s') = R_{step} + R_{event} + R_{shaping}$$

| Kejadian (*Event*) | Nilai Reward | Alasan & Justifikasi Desain |
| :--- | :---: | :--- |
| **Mencapai Goal** | $+60.0$ | Reward terminal positif besar untuk memprioritaskan penyelesaian misi pengantaran. |
| **Langkah Valid** | $-0.2$ | Penalti waktu konstan untuk mendorong efisiensi dan lintasan terpendek. |
| **Menabrak Rak / Dinding** | $-5.0$ | Penalti tabrakan substansial untuk menghalangi gerakan membentur rintangan. |
| **Melebihi Batas Langkah (Timeout)** | $-10.0$ | Penalti terminal jika gagal mencapai tujuan dalam batas langkah maksimal ($120$ langkah). |
| **Reward Shaping Jarak** | $+0.5 \times (\text{dist}_{prev} - \text{dist}_{curr})$ | *Potential-based reward shaping* yang memberikan dorongan positif jika mendekati tujuan dan penalti jika menjauhi tujuan. |

### 5.6 Terminal Condition
Episode dihentikan (*done = True*) apabila salah satu dari dua kondisi terpenuhi:
1. Koordinat robot sama dengan koordinat tujuan ($p_{robot} = p_{goal}$).
2. Jumlah langkah mencapai batas maksimal ($step \ge 120$).

---

## 6. Metodologi DQN

### 6.1 Arsitektur Q-Network
Estimasi nilai $Q(s, a)$ dimodelkan menggunakan jaringan saraf tiruan Multi-Layer Perceptron (MLP) dengan parameter:
- **Input Layer**: 19 neuron (sesuai dimensi vektor observasi).
- **Hidden Layer 1**: 128 neuron, aktivasi ReLU.
- **Hidden Layer 2**: 128 neuron, aktivasi ReLU.
- **Hidden Layer 3**: 64 neuron, aktivasi ReLU.
- **Output Layer**: 4 neuron linear (masing-masing merepresentasikan estimasi $Q$-value untuk aksi $\{0, 1, 2, 3\}$).

### 6.2 Replay Buffer
Pengalaman agent disimpan dalam struktur buffer antrean melingkar (*circular deque*) berkapasitas $40.000$ tupel transisi:

$$e_t = (s_t, a_t, r_t, s_{t+1}, d_t)$$

Ketika ukuran buffer telah mencapai batas batch minimum ($64$), dilakukan penarikan sampel acak seragam (*uniform sampling*) untuk memutus korelasi temporal antar sampel berurutan (*decorrelation of sequential observations*).

### 6.3 Epsilon-Greedy Exploration
Kebijakan eksplorasi-eksploitasi diatur oleh jadwal peluruhan eksponensial:

$$\epsilon_{t+1} = \max(\epsilon_{min}, \epsilon_t \times \lambda)$$

- $\epsilon_{start} = 1.0$ (eksplorasi acak total pada awal pelatihan).
- $\epsilon_{min} = 0.05$ (eksplorasi residual 5% untuk mencegah kejenuhan kebijakan).
- $\lambda = 0.994$ (faktor peluruhan per episode).

### 6.4 Target Network & Stabilisasi
Untuk mencegah pergeseran target estimasi (*moving target problem*), digunakan dua jaringan identik: *Policy Network* $\theta$ dan *Target Network* $\theta^{-}$. Target Bellman dihitung melalui:

$$y_i = r_i + \gamma \max_{a'} Q(s'_i, a'; \theta^{-}) (1 - d_i)$$

Pembaruan bobot jaringan target dilakukan menggunakan *soft Polyak update* setiap interval 10 langkah:

$$\theta^{-} \leftarrow \tau \theta + (1 - \tau) \theta^{-}, \quad \text{dengan } \tau = 0.005$$

### 6.5 Loss Function dan Optimizer
- **Loss Function**: Huber Loss (Smooth L1 Loss) yang kurang sensitif terhadap outlier gradien dibandingkan MSE:

$$L(\theta) = \frac{1}{B} \sum_{i=1}^B \text{Smooth}_{L1} \left( Q(s_i, a_i; \theta) - y_i \right)$$

- **Optimizer**: Adam dengan learning rate $\alpha = 0.0005$.
- **Gradient Clipping**: Gradien dipangkas pada batas norma maksimum $5.0$ untuk stabilitas numerik.

### 6.6 Ringkasan Hyperparameter

| Parameter | Nilai Konfigurasi | Keterangan |
| :--- | :---: | :--- |
| Learning Rate ($\alpha$) | $5 \times 10^{-4}$ | Laju pembaruan Adam |
| Discount Factor ($\gamma$) | $0.99$ | Pembobotan reward masa depan |
| Replay Buffer Size | $40.000$ | Kapasitas transisi memori |
| Mini-Batch Size | $64$ | Jumlah sampel per pembaruan gradien |
| Total Episodes | $600$ | Jumlah episode pelatihan penuh |
| Max Steps / Episode | $120$ | Ambang batas timeout |
| $\epsilon_{start}$ / $\epsilon_{min}$ / $\epsilon_{decay}$ | $1.0$ / $0.05$ / $0.994$ | Parameter jadwal eksplorasi |
| Target Update $\tau$ | $0.005$ | Koefisien soft Polyak target network |
| Target Update Interval | $10$ langkah | Frekuensi pembaruan jaringan target |
| Random Seed | $42$ | Seed kontrol reproduktibilitas |

---

## 7. Baseline Pembanding

Dua algoritma pencarian graf klasik diimplementasikan dalam `algorithms/pathfinding.py`:

### 7.1 Breadth-First Search (BFS)
- **Mekanisme**: Penelusuran ruang keadaan tingkat-demi-tingkat (*level-order traversal*) menggunakan antrean FIFO.
- **Karakteristik**: Menjamin penemuan jalur terpendek mutlak pada graf berbobot seragam (*unweighted grid*).
- **Asumsi**: Memiliki akses langsung ke seluruh matriks rintangan secara global.

### 7.2 A\* Search (A-Star)
- **Mekanisme**: Algoritma pencarian berheuristik menggunakan antrean prioritas (*min-heap*), di mana biaya dievaluasi melalui $f(n) = g(n) + h(n)$.
- **Fungsi Heuristik**: Menggunakan jarak Manhattan yang terbukti bersifat *admissible* dan *consistent* untuk grid 4-arah:

$$h(n) = |r_n - r_{goal}| + |c_n - c_{goal}|$$

- **Karakteristik**: Menjamin panjang lintasan optimal serupa BFS, namun mengeksplorasi simpul (*nodes expanded*) jauh lebih sedikit pada lingkungan yang memiliki koridor dan rintangan kompleks.

---

## 8. Desain Eksperimen

### 8.1 Protokol Pelatihan
- Model dilatih pada skenario **Moderate Map** (12x12) selama **600 episode** menggunakan random seed 42.
- Riwayat metrik (reward, loss, steps, tabrakan, success, epsilon) dicatat per episode ke file `dataset/training_history.csv`.
- Checkpoint model terbaik disimpan secara dinamis ke `models/best_model.pth` berdasarkan nilai moving average success rate dan reward tertinggi (diperoleh checkpoint terbaik pada rata-rata reward $+64.9$).

### 8.2 Protokol Evaluasi
- Evaluasi dilakukan secara terpisah dari fase pelatihan (`training/evaluate.py`).
- Eksplorasi dinonaktifkan ($\epsilon = 0.0$ atau *pure greedy policy*) untuk menguji performa deterministik.
- Pengujian dijalankan sebanyak **30 episode independen per skenario** pada 4 skenario denah.
- BFS dan A\* dievaluasi pada posisi start dan goal yang identik untuk setiap denah.
- Seluruh metrik disimpan ke `dataset/evaluation_results.csv`.

---

## 9. Hasil Training

Pelatihan berjalan selama **42,18 detik** pada prosesor CPU. Perkembangan metrik pembelajaran tercatat sebagai berikut:

| Episode Interval | Epsilon ($\epsilon$) | Rata-rata Reward (50 Ep) | Success Rate (50 Ep) | Rata-rata Langkah | Rata-rata Tabrakan | Huber Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1–50 | $0.994 \to 0.740$ | $-160.0$ | $32.0\%$ | $116$ | $28$ | $0.2516$ |
| 51–100 | $0.740 \to 0.548$ | $-9.4$ | $94.0\%$ | $31$ | $5$ | $0.0850$ |
| 101–150 | $0.548 \to 0.405$ | $+33.2$ | **$100.0\%$** | $42$ | $2$ | $0.0903$ |
| 151–200 | $0.405 \to 0.300$ | $+48.0$ | **$100.0\%$** | $35$ | $1$ | $0.0234$ |
| 201–300 | $0.300 \to 0.164$ | $+56.6$ | **$100.0\%$** | $27$ | $1$ | $0.1400$ |
| 301–450 | $0.164 \to 0.067$ | $+63.0$ | **$100.0\%$** | $26$ | $0$ | $0.0399$ |
| 451–600 | $0.067 \to 0.050$ | **$+64.9$** | **$100.0\%$** | **$22$** | **$0$** | $0.1852$ |

### 9.1 Visualisasi Metrik Pelatihan

Berikut adalah visualisasi telemetri proses pelatihan model DQN:

![Training Summary](visualizations/training_summary.png)

*Gambar 1: Dashboard Komprehensif Metrik Pelatihan DQN: (A) Progresi Reward Kumulatif dan Moving Average, (B) Konvergensi Huber Loss Jaringan Kebijakan, (C) Perkembangan Success Rate, dan (D) Penurunan Epsilon Eksplorasi terhadap Langkah Episode.*

Detail grafik individu:
- **Reward Progression**: `visualizations/training_reward.png`
- **Huber Loss Convergence**: `visualizations/training_loss.png`
- **Success Rate Curve**: `visualizations/success_rate.png`
- **Epsilon Schedule**: `visualizations/epsilon_decay.png`

### 9.2 Interpretasi Grafik Pelatihan
1. **Fase Eksplorasi Awal (Ep 1–50)**: Reward berada pada angka negatif (sekitar $-260$ hingga $-160$) dengan tingkat tabrakan tinggi (28–46 tabrakan per episode). Pada fase ini, agent mengumpulkan transisi acak untuk mengisi replay buffer.
2. **Fase Akselerasi Pembelajaran (Ep 51–150)**: Terjadi kenaikan performa drastis. Tingkat keberhasilan melonjak dari 32% ke 100% pada episode 150. Hal ini menandakan bahwa jaringan telah berhasil memetakan bahwa menabrak rintangan memberikan penalti besar sedangkan mendekati dan menyentuh goal memberikan penguatan positif yang masif.
3. **Fase Optimasi Lintasan (Ep 151–600)**: Setelah success rate stabil di 100%, agent terus memangkas langkah yang tidak perlu. Jumlah langkah rata-rata per episode berkurang dari 42 langkah menjadi tepat **22 langkah** (yang merupakan rute terpendek absolut pada grid 12x12 tersebut), dengan **0 tabrakan**.

---

## 10. Hasil Evaluasi

Berikut adalah tabel hasil evaluasi empiris formal (berdasarkan file aktual `dataset/evaluation_results.csv`):

| Skenario Lingkungan | Metode | Jumlah Ep | Success Rate | Rata-rata Reward | Panjang Jalur | Langkah Selesai | Rata-rata Tabrakan | Rasio Tabrakan/Langkah | Rata-rata Latensi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple Map** (10x10) | **BFS** | 1 | 100.0% | - | 18.0 | 18.0 | 0.0 | 0.0000 | 0.109 ms |
| | **A\*** | 1 | 100.0% | - | 18.0 | 18.0 | 0.0 | 0.0000 | 0.138 ms |
| | **DQN** | 30 | **100.0%** | $+65.40 \pm 0.0$ | **18.0** | **18.0** | **0.0** | **0.0000** | **0.153 ms** |
| **Moderate Map** (12x12) | **BFS** | 1 | 100.0% | - | 22.0 | 22.0 | 0.0 | 0.0000 | 0.113 ms |
| | **A\*** | 1 | 100.0% | - | 22.0 | 22.0 | 0.0 | 0.0000 | 0.152 ms |
| | **DQN** | 30 | **100.0%** | $+66.60 \pm 0.0$ | **22.0** | **22.0** | **0.0** | **0.0000** | **0.150 ms** |
| **Complex Map** (12x12) | **BFS** | 1 | 100.0% | - | 20.0 | 20.0 | 0.0 | 0.0000 | 0.111 ms |
| | **A\*** | 1 | 100.0% | - | 20.0 | 20.0 | 0.0 | 0.0000 | 0.107 ms |
| | **DQN** | 30 | **0.0%** | $-519.5 \pm 0.0$ | 120.0 | 120.0 | 99.0 | 0.8250 | 0.151 ms |
| **Generalization Map** (12x12) | **BFS** | 1 | 100.0% | - | 22.0 | 22.0 | 0.0 | 0.0000 | 0.126 ms |
| | **A\*** | 1 | 100.0% | - | 22.0 | 22.0 | 0.0 | 0.0000 | 0.170 ms |
| | **DQN** | 30 | **0.0%** | $-524.0 \pm 0.0$ | 120.0 | 120.0 | 100.0 | 0.8333 | 0.170 ms |

### 10.1 Perbandingan Visual dengan Baseline

![Baseline Comparison](visualizations/baseline_comparison.png)

*Gambar 2: Perbandingan Panjang Jalur Navigasi Antara DQN, BFS, dan A\* pada Seluruh Skenario.*

![Evaluation Metrics](visualizations/evaluation_metrics.png)

*Gambar 3: Multi-Metric Benchmark: (A) Success Rate, (B) Panjang Jalur, (C) Latensi Komputasi per Langkah, dan (D) Rata-rata Tabrakan.*

---

## 11. Visualisasi Navigasi

### 11.1 Peta Denah Gudang
Peta denah gudang logistik berukuran 12x12 yang digunakan dalam eksperimen utama:

![Warehouse Map](visualizations/warehouse_map.png)

*Gambar 4: Visualisasi Denah Gudang 12x12: Rak Penyimpanan (abu-abu gelap), Lorong Bebas (putih keabu-abuan), Titik Start (hijau toska), dan Titik Tujuan / Drop-off Bay (merah).*

### 11.2 Jalur Navigasi DQN vs BFS vs A\*
Perbandingan lintasan aktual yang ditempuh robot disajikan pada gambar berikut:

![DQN Learned Path](visualizations/dqn_path.png)

*Gambar 5: Overlay Lintasan Navigasi pada Moderate Map (12x12): Lintasan DQN (garis cyan tebal dengan titik koordinat) bergerak mulus melewati lorong dan tiba tepat di tujuan, berhimpitan secara sempurna dengan lintasan optimal BFS dan A\*.*

### 11.3 Animasi Pergerakan Robot
Animasi pergerakan langkah-demi-langkah robot AGV dari titik awal hingga mencapai titik bongkar muat disimpan dalam format GIF resolusi tinggi:

![Robot Navigation GIF](visualizations/robot_navigation.gif)

*Gambar 6: Animasi Simulasi Navigasi AGV Mandiri (28 Frame) Menelusuri Lorong Gudang Bebas Tabrakan Menuju Titik Tujuan.*

---

## 12. Pembahasan

### 12.1 Apakah Agent Benar-benar Mempelajari Navigasi?
**Ya, secara meyakinkan.** Evaluasi menunjukkan bahwa agent DQN tidak bergerak secara acak atau menggunakan rute hardcoded:
- Pada awal pelatihan (Ep 1–25), agent sering terjebak di sudut dinding dan mengalami rata-rata 46 tabrakan.
- Seiring bertambahnya episode dan penarikan sampel dari replay buffer, agent secara progresif mempelajari gradien spasial dan koordinasi aksi: menghindari sel rak dan memilih arah yang meminimalkan jarak ke target.
- Pada episode akhir dan fase evaluasi, agent menuntaskan tugas navigasi dalam **22 langkah tepat**, menghasilkan lintasan paling efisien yang secara matematis dimungkinkan pada denah tersebut.

### 12.2 Kondisi Keberhasilan dan Kegagalan
- **Kondisi Berhasil**: 
  1. *Moderate Map*: Topologi gudang teratur dengan lorong vertikal dan persilangan horizontal terbuka. Agent berhasil 100% dengan 0 tabrakan.
  2. *Simple Map*: Topologi terbuka dengan rintangan minimal. Agent yang dilatih pada Moderate Map mampu melakukan *zero-shot transfer* ke Simple Map dengan **100% keberhasilan** dan panjang jalur 18 langkah (identik dengan rute optimal A\*).
- **Kondisi Gagal**:
  1. *Complex Map* & *Generalization Map*: Agent mengalami kegagalan (0% success rate, 99–100 tabrakan akibat timeout 120 langkah).
  2. **Penyebab Kegagalan**: Pada skenario Complex Map terdapat bentuk rintangan non-konveks (*concave dead-ends* / bentuk U). Karena vektor observasi hanya memuat sensor lokal $3 \times 3$ dan jarak ray-casting 4 arah (bukan peta graf global), dorongan gradien jarak Manhattan menarik agent ke dinding partisi yang menghadap ke arah goal, menyebabkan agent terjebak dalam *local minimum* (bergerak bolak-balik di depan rintangan). BFS dan A\* berhasil 100% karena keduanya memiliki visibilitas graf menyeluruh dan algoritma *backtracking*.

### 12.3 Perbandingan Efisiensi Jalur dan Latensi
- **Efisiensi Rute**: Pada skenario yang berhasil, efisiensi jalur DQN adalah **100%** (22 langkah vs 22 langkah A\* pada Moderate Map; 18 langkah vs 18 langkah A\* pada Simple Map).
- **Kecepatan Inferensi**: 
  - Waktu komputasi A\* adalah $0,152 \text{ ms}$ untuk melakukan pencarian graf menyeluruh.
  - Waktu inferensi DQN adalah **$0,150 \text{ ms}$** per evaluasi feedforward neural network.
  - Keunggulan utama DQN terletak pada sifatnya yang *constant-time $O(1)$ inference* per langkah gerak, tidak mengalami ledakan eksponensial kompleksitas pencarian ruang keadaan seperti algoritma penelusuran graf murni jika ukuran lingkungan membesar menjadi ratusan meter.

---

## 13. Keterbatasan

1. **Skala Peta Grid Statis**: Eksperimen difokuskan pada grid diskret $10 \times 10$ dan $12 \times 12$. Pada grid yang jauh lebih besar ($50 \times 50$ atau lebih), vektor observasi tanpa komponen memori jangka panjang (*Recurrent Neural Network* / LSTM) akan semakin rentan mengalami jebakan jalan buntu.
2. **Ketiadaan Rintangan Dinamis**: Rintangan yang dimodelkan saat ini bersifat statis (rak gudang). Keberadaan pekerja gudang atau AGV lain yang bergerak belum dimasukkan dalam iterasi dasar ini.
3. **Representasi Vektor vs Matriks Citra Peta**: Penggunaan vektor fitur 19 dimensi sangat cepat dan ringan pada CPU, namun kehilangan detail topologi global non-lokal yang biasanya dapat ditangkap oleh *Convolutional Neural Network* (CNN) yang membaca keseluruhan *occupancy grid map*.

---

## 14. Kesimpulan

Project *Autonomous Warehouse Navigation with Deep Q-Learning* telah berhasil dirancang, diimplementasikan, dilatih, diverifikasi, dan didokumentasikan secara komprehensif:
1. Agent DQN berhasil dilatih dari interaksi murni dalam 600 episode (durasi 42 detik pada CPU), mencapai tingkat keberhasilan **100%** dan **0 tabrakan** pada lingkungan gudang target.
2. Jalur yang dipelajari DQN **sempurna menyamai jalur optimal algoritma A\* dan BFS** (22 langkah pada Moderate Map dan 18 langkah pada Simple Map).
3. Evaluasi jujur dan objektif mengungkapkan bahwa kebijakan reaktif berbasis sensor lokal rentan terhadap jebakan jalan buntu pada labirin kompleks, menunjukkan batasan mendasar antara pendekatan pencarian graf global versus pendekatan pembelajaran berbasis nilai aksi lokal.
4. Seluruh kode, model checkpoint, tabel metrik, visualisasi resolusi tinggi, animasi navigasi, serta aplikasi Streamlit telah berfungsi penuh dan dapat direproduksi secara mandiri.

---

## 15. Pengembangan Selanjutnya

Untuk pengembangan fase lanjutan pada sistem operasional skala industri:
1. **Peningkatan Arsitektur (Dueling Double DQN & DRQN)**: Mengintegrasikan *Dueling DQN* untuk memisahkan estimasi *State-Value* $V(s)$ dan *Advantage* $A(s, a)$, serta lapisan LSTM (*Deep Recurrent Q-Network*) untuk mengatasi keterbatasan observasi parsial (*POMDP*).
2. **Dynamic Obstacles**: Memasukkan rintangan bergerak (manusia atau forklift lain) dengan menambahkan fitur kecepatan relatif ke dalam vektor observasi.
3. **Multi-Agent Warehouse Navigation (MAPPO / QMIX)**: Memperluas lingkungan ke skenario multi-robot dengan protokol koordinasi desentralisasi untuk menghindari kemacetan di persimpangan lorong (*aisle intersection deadlocks*).
4. **Continuous Action Space (PPO / SAC)**: Mentransisikan model dari grid diskret ke kinematika robot diferensial nyata (kecepatan linear $v$ dan kecepatan sudut $\omega$).

---

## 16. Cara Menjalankan Project

### 16.1 Persyaratan Sistem
- Python 3.10 hingga 3.15
- Sistem Operasi: Windows, Linux, atau macOS

### 16.2 Instalasi Dependensi
```bash
# Clone atau buka direktori project
cd "Autonomous Warehouse Navigation with Deep Q-Learning"

# Membuat virtual environment (opsional namun disarankan)
python -m venv .venv

# Aktivasi virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Instalasi library yang dibutuhkan
pip install -r requirements.txt
```

### 16.3 Menjalankan Perintah Utama
Proyek ini menyediakan CLI terpadu via `main.py`:

```bash
# 1. Menjalankan training DQN (600 episode)
python main.py --mode train --map moderate_map --episodes 600

# 2. Menjalankan evaluasi komparatif lengkap (DQN vs BFS vs A*)
python main.py --mode evaluate

# 3. Menghasilkan seluruh visualisasi (peta, kurva training, perbandingan, & animasi GIF)
python main.py --mode visualize

# 4. Menjalankan seluruh alur end-to-end secara berurutan
python main.py --mode all
```

### 16.4 Menjalankan Dashboard Interaktif Streamlit
```bash
python -m streamlit run app/streamlit_app.py --theme.base dark
```
Akses dashboard di browser melalui alamat: `http://localhost:8501`.

---

## 17. Daftar Artefak

### 17.1 Model Tersimpan
- `models/best_model.pth`: Checkpoint bobot neural network terbaik (berdasarkan moving average success rate dan reward tertinggi).
- `models/final_model.pth`: Bobot neural network pada episode terakhir (episode 600).

### 17.2 Dataset & Log Eksperimen
- `dataset/training_history.csv`: Log per episode (episode, reward, steps, collisions, success, epsilon, loss, moving averages).
- `dataset/evaluation_results.csv`: Tabel ringkasan hasil evaluasi deterministik DQN vs BFS vs A\* pada seluruh skenario.
- `results/experiment_config.json`: Metadata konfigurasi lengkap eksperimen.

### 17.3 Visualisasi Grafis (`visualizations/`)
- `visualizations/warehouse_map.png`: Peta denah gudang logistik resolusi tinggi.
- `visualizations/training_reward.png`: Kurva reward per episode dan moving average 50 episode.
- `visualizations/training_loss.png`: Kurva konvergensi Huber loss.
- `visualizations/success_rate.png`: Kurva tingkat keberhasilan mencapai target.
- `visualizations/epsilon_decay.png`: Jadwal peluruhan eksplorasi $\epsilon$-greedy.
- `visualizations/training_summary.png`: Dashboard komposit 4 panel seluruh metrik training.
- `visualizations/baseline_comparison.png`: Grafik batang perbandingan panjang jalur DQN vs BFS vs A\*.
- `visualizations/evaluation_metrics.png`: Grafik komparasi multi-metrik (Success, Length, Latency, Collisions).
- `visualizations/dqn_path.png`: Visualisasi lintasan navigasi DQN dan baseline pada grid gudang.
- `visualizations/robot_navigation.gif`: Animasi GIF 28 frame navigasi mandiri AGV.

### 17.4 Tangkapan Layar Aplikasi Streamlit (`screenshots/`)
Seluruh tangkapan layar diambil secara otomatis menggunakan Microsoft Edge via Chrome DevTools Protocol (CDP) dengan *Dynamic Viewport Extension* dalam format dark mode profesional:
- `screenshots/01_overview.png`: Tangkapan layar penuh Tab *System Overview*.
- `screenshots/02_warehouse_simulator.png`: Tangkapan layar penuh Tab *Warehouse Simulator*.
- `screenshots/03_training_diagnostics.png`: Tangkapan layar penuh Tab *Training Diagnostics*.
- `screenshots/04_baseline_benchmark.png`: Tangkapan layar penuh Tab *Baseline Benchmark*.
- `screenshots/05_scenario_testing.png`: Tangkapan layar penuh Tab *Scenario Stress Testing*.
- `screenshots/SCREENSHOT-INDEX.md`: Indeks dokumentasi inventarisasi tangkapan layar.

---

## 18. Referensi

1. Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2015). *Human-level control through deep reinforcement learning*. **Nature**, 518(7540), 529-533.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press, Cambridge, MA.
3. Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under reward transformations: Theory and application to reward shaping*. **ICML**, 99, 278-287.
4. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). *A formal basis for the heuristic determination of minimum cost paths*. **IEEE Transactions on Systems Science and Cybernetics**, 4(2), 100-107.
5. Wurman, P. R., D'Andrea, R., & Mountz, M. Q. (2008). *Coordinating hundreds of cooperative, autonomous vehicles in warehouses*. **AI Magazine**, 29(1), 9-20.
