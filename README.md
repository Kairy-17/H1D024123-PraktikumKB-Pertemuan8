# Pertemuan 8: Jaringan Syaraf Tiruan (JST) 3 - Klasifikasi Gambar Rock-Paper-Scissors

## 📝 Deskripsi Tugas & Studi Kasus
Praktikum ini berfokus pada implementasi konsep *Convolutional Neural Network* (CNN) untuk melakukan klasifikasi gambar tiga bentuk isyarat tangan, yaitu Gunting (*Scissors*), Batu (*Rock*), dan Kertas (*Paper*). Model dilatih untuk mengekstrak fitur spasial unik dari gambar agar mampu mengenali kelas gambar secara optimal menggunakan dataset lokal yang diekstrak ke dalam folder `./rockpaperscissors`.

## 🧠 Arsitektur Model CNN
Model dibangun menggunakan struktur **Sequential** melalui library **TensorFlow** dan **Keras** dengan susunan layer sebagai berikut:
* **Conv2D Layer 1**: 32 filters, kernel size (3,3), fungsi aktivasi `ReLU`, menerima input shape gambar `(150, 150, 3)`.
* **MaxPooling2D 1**: Ukuran filter (2,2) untuk mereduksi dimensi spasial data gambar.
* **Conv2D Layer 2**: 64 filters, kernel size (3,3), fungsi aktivasi `ReLU`.
* **MaxPooling2D 2**: Ukuran filter (2,2).
* **Conv2D Layer 3**: 128 filters, kernel size (3,3), fungsi aktivasi `ReLU`.
* **MaxPooling2D 3**: Ukuran filter (2,2).
* **Flatten Layer**: Mengubah feature map hasil pooling menjadi vektor satu dimensi (1D).
* **Dense Layer (Hidden)**: 512 neuron dengan fungsi aktivasi `ReLU`.
* **Dense Layer (Output)**: 3 neuron dengan fungsi aktivasi `Softmax` untuk klasifikasi multi-kelas (*Rock, Paper, Scissors*).

## 📊 Pengolahan Gambar & Alur Kerja
1. **Dataset Preparation**: Menggunakan dataset gambar tangan yang terbagi ke dalam sub-folder kelas masing-masing (`paper`, `rock`, `scissors`).
2. **ImageDataGenerator**: Manajemen pemuatan data gambar otomatis dengan melakukan normalisasi nilai piksel (`rescale=1/255`) serta pemisahan data latih dan validasi otomatis dengan rasio **80% data training** dan **20% data validation** menggunakan parameter `validation_split`.
3. **Kompilasi Model**: Mengonfigurasi model dengan loss function `categorical_crossentropy` dan `Adam` optimizer selama 10 epoch.
4. **Evaluasi**: Mengukur tingkat performa model berdasarkan nilai akhir *Validation Loss* dan *Validation Accuracy* serta menampilkan tren grafiknya menggunakan `matplotlib`.

## 🚀 Cara Menjalankan Program

Salin dan ikuti seluruh baris perintah di dalam satu kotak di bawah ini secara berurutan:

```bash
# LANGKAH 1: Install seluruh library pendukung yang diperlukan di terminal
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn

# LANGKAH 2: Pastikan folder dataset bernama 'rockpaperscissors' hasil ekstrak
# sudah diletakkan di dalam folder yang sama dengan file skrip 'cnn_rps.py'

# LANGKAH 3: Jalankan program utama CNN Anda melalui terminal
python cnn_rps.py
