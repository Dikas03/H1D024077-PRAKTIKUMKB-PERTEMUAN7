# Jaringan Syaraf Tiruan 2 - Klasifikasi Bunga Iris

Repositori ini berisi implementasi praktikum Kecerdasan Buatan (KB) Pertemuan 7 mengenai **Jaringan Syaraf Tiruan 2**. Program ini mengimplementasikan model *Artificial Neural Network* (ANN) menggunakan TensorFlow dan Keras untuk mengklasifikasikan spesies bunga Iris.

## 📌 Tujuan Praktikum
Menerapkan konsep Jaringan Syaraf Tiruan (JST) dalam kode Python menggunakan library TensorFlow dan Keras.

## 🛠️ Alat dan Bahan
- Python 3.x
- Library: `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

## 📖 Deskripsi Kode
Sistem JST yang dibangun terdiri atas:
- **Input Layer:** Menerima 4 fitur (*sepal length, sepal width, petal length, petal width*).
- **Hidden Layers:** Tiga *Dense layer* dengan masing-masing 1000, 500, dan 300 neuron. Fungsi aktivasi menggunakan ReLU.
- **Output Layer:** *Dense layer* dengan 3 neuron (mewakili 3 kelas bunga: *Setosa, Versicolor, Virginica*). Fungsi aktivasi menggunakan Softmax.
- **Optimizer & Loss:** Menggunakan optimizer `Adam` dan loss function `sparse_categorical_crossentropy`.
- **Training:** Model dilatih selama 50 epoch dengan *batch size* 32.

## 🚀 Cara Menjalankan Program

1.  **Clone Repositori (Jika sudah di GitHub)**
    ```bash
    git clone https://github.com/[USERNAME_GITHUB_ANDA]/[NIM]-PraktikumKB-Pertemuan7.git
    cd [NIM]-PraktikumKB-Pertemuan7
    ```

2.  **Install Dependencies (Library yang dibutuhkan)**
    Pastikan Anda sudah menginstal TensorFlow dan library data science lainnya:
    ```bash
    pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **Eksekusi Program**
    Jalankan *script* utama:
    ```bash
    python praktikum7.py
    ```

4.  **Alur Eksekusi Program:**
    - Program akan mengunduh dataset secara otomatis.
    - Model akan di-*compile* dan dilatih (*training*) dengan menampilkan proses per-*epoch*.
    - Grafik pergerakan *loss* dan *accuracy* akan dimunculkan (tutup jendela grafik untuk melanjutkan program).
    - Matriks Konfusi (*Confusion Matrix*) akan ditampilkan dalam bentuk *heatmap* (tutup jendela grafik untuk melanjutkan).
    - Terakhir, program akan meminta input manual dari *user* untuk memprediksi data baru secara interaktif.

## 📊 Hasil Percobaan

### 1. Evaluasi Model (Data Testing)
Berdasarkan hasil pelatihan (*training*) 50 epoch pada dataset Iris, model berhasil dievaluasi dengan hasil yang sangat baik:
- **Loss:** `0.0999`
- **Accuracy:** `0.9666` (96.67%)

### 2. Grafik Training History
*(Grafik pergerakan metrik Loss dan Accuracy dari epoch pertama hingga epoch ke-50)*

![Training History](figure/Training%20history.png)

### 3. Confusion Matrix
*(Hasil pemetaan prediksi model dibandingkan dengan label kelas aslinya pada data validasi/testing)*

![Confusion Matrix](figure/Confusion%20Matrix.png)

### 4. Hasil Prediksi Interaktif (Test Case)
Ketika program diuji coba dengan data ukuran bunga Iris baru:
- Sepal length: `5.1`
- Sepal width: `3.5`
- Petal length: `1.4`
- Petal width: `0.2`

**Output Prediksi:** `Iris-setosa` (Berhasil diklasifikasikan dengan tepat!)

## 📂 Struktur Repositori
- `praktikum7.py` : Berisi source code utama dari praktikum.
- `README.md`     : File dokumentasi petunjuk program.
- `iris/`         : Folder berisi offline dataset Iris.
- `figure/`       : Folder berisi grafik *Training History* dan *Confusion Matrix*.

---
*Tugas ini diselesaikan dalam rangka memenuhi Praktikum Pertemuan 7.*
