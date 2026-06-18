# Loan Prediction CI/CD Workflow (MLflow & Docker)

Repositori ini berisi workflow CI/CD otomatis untuk pelatihan ulang model klasifikasi persetujuan pinjaman (*Loan Approval*) menggunakan **MLflow** dan **Docker**. Workflow ini dirancang untuk memastikan integrasi berkelanjutan pada model machine learning (M LOps) dari tahap eksperimen, pelacakan metrik, hingga pembentukan kontainer Docker siap-pakai.

---

## 📂 Struktur Proyek

```text
├── .github/
│   └── workflows/
│       └── ci.yml             # Workflow GitHub Actions untuk CI & Docker Build/Push
├── MLProject/
│   ├── MLProject              # File konfigurasi MLflow Project descriptor
│   ├── conda.yaml             # Spesifikasi dependensi untuk environment
│   ├── dataset_clean.csv      # Dataset bersih untuk pelatihan model
│   ├── docker_hub_link.txt    # Tautan ke repositori Docker Hub resmi
│   └── modelling.py           # Skrip utama pelatihan ulang model RandomForest
├── .gitignore                 # Konfigurasi pengabaian file tidak diperlukan git
└── README.md                  # Dokumentasi proyek (file ini)
```

---

## 🚀 Fitur Utama

1. **Pelatihan Ulang Model Otomatis**: Melatih model `RandomForestClassifier` menggunakan dataset [dataset_clean.csv](file:///c:/Users/Julianda/loan-prediction-ci-workflow/MLProject/dataset_clean.csv).
2. **Pelacakan Eksperimen Terintegrasi**: Menggunakan **MLflow** untuk melacak parameter (`n_estimators`, `max_depth`), metrik evaluasi (`accuracy`, `precision`, `recall`, `f1_score`), serta menyimpan artefak model.
3. **Koneksi Remote DagsHub**: Terhubung secara otomatis ke remote MLflow tracking server di DagsHub ketika dijalankan secara manual di luar workflow CI.
4. **Visualisasi Evaluasi**: Menghasilkan plot evaluasi secara dinamis berupa *Confusion Matrix* dan *Feature Importance*, kemudian mengunggahnya langsung ke MLflow Run.
5. **CI/CD & Dockerization**:
   - Menjalankan pengujian otomatis setiap ada *push* ke cabang `main` atau `master`.
   - Mengambil lokasi keluaran model terbaru secara dinamis.
   - Mengompilasi kontainer Docker resmi berisi model MLflow menggunakan perintah `mlflow models build-docker`.
   - Mengunggah Docker Image ke [Docker Hub](https://hub.docker.com/r/haikal1012/loan-model) secara otomatis (jika kredensial rahasia tersedia).

---

## 🛠️ Cara Menjalankan Proyek Secara Lokal

### Prasyarat
Pastikan Anda telah memasang Python 3.12 dan pustaka yang dibutuhkan. Anda bisa memasang dependensi secara manual melalui pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn mlflow scikit-optimize dagshub
```

### Opsi 1: Menjalankan Skrip Python Langsung
Anda dapat langsung mengeksekusi skrip pemodelan. Opsi ini akan otomatis mencoba terhubung dengan DagsHub untuk pencatatan eksperimen secara remote:

```bash
python MLProject/modelling.py --n-estimators 100 --max-depth 10
```

### Opsi 2: Menjalankan Menggunakan MLflow Project CLI
MLflow memfasilitasi reprodusibilitas melalui file `MLProject`. Jalankan perintah berikut untuk mengeksekusi model dalam lingkungan lokal Anda:

```bash
mlflow run MLProject/ --entry-point main --env-manager local -P n_estimators=100 -P max_depth=10
```

---

## 🤖 Workflow CI (GitHub Actions)

Workflow didefinisikan dalam [.github/workflows/ci.yml](file:///c:/Users/Julianda/loan-prediction-ci-workflow/.github/workflows/ci.yml) dengan tahapan sebagai berikut:

1. **Checkout Code**: Mengambil seluruh repositori aktif.
2. **Set up Python**: Mengonfigurasi lingkungan Python 3.12 di runner GitHub Actions.
3. **Install Dependencies**: Memasang paket python yang diperlukan (`mlflow`, `scikit-learn`, dll.).
4. **Run MLflow Project**: Melatih model dengan menjalankan perintah `mlflow run`.
5. **Upload Artifacts**: Mengunggah file `mlruns/` (log eksperimen) serta plot gambar evaluasi ke GitHub Artifacts agar dapat diunduh.
6. **Build Docker Image**:
   - Memindai file `MLmodel` yang dihasilkan untuk mendapatkan path spesifik model.
   - Menjalankan `mlflow models build-docker` untuk mengemas model ke dalam kontainer Linux siap pakai.
7. **Push to Docker Hub**: Jika rahasia `DOCKER_USERNAME` dan `DOCKER_PASSWORD` terdaftar di GitHub Secrets, image kontainer akan dipublikasikan ke Docker Hub:
   - Tautan Resmi: [haikal1012/loan-model](https://hub.docker.com/r/haikal1012/loan-model)