# Kecerdasan-Artifisial

## Perubahan yang terjadi

### **1. Pembersihan dan Pemrosesan Data:**
   - **Fungsi `preprocess_serving_grams`**: 
     - Mengonversi nilai dalam kolom `Serving`, `Grams`, dan `Calories` ke dalam format numerik yang benar.
     - Menghapus **pemisa ribuan** (jika ada) dari kolom `calories` untuk memastikan angka dibaca dengan benar.
     - Menggunakan **titik (`.`)** sebagai pemisah desimal dan **menghapus koma (`,`)** yang mungkin digunakan sebagai pemisah ribuan.
   
   - **Perubahan yang diterapkan:**
     - Menambahkan fungsi untuk mengekstrak **`serving_numeric`**, **`grams_in_serving`**, dan **`grams_numeric`**.
     - **`calories`** dibaca dengan benar setelah menghapus koma.

### **2. Penambahan Fitur:**
   - **Fungsi `add_features`**:
     - Menambahkan fitur-fitur tambahan untuk model regresi linear:
       - **`interaction_term`**: Hasil perkalian antara `serving_numeric` dan `grams_numeric`.
       - **`serving_squared`**: Kuadrat dari `serving_numeric`.
       - **`grams_squared`**: Kuadrat dari `grams_numeric`.
       - **`interaction_log`**: Logaritma dari `grams_numeric + 1` (untuk mencegah log(0)).

### **3. Regresi Linear Manual:**
   - **Kelas `LinearRegressionManual`**:
     - Implementasi regresi linear manual tanpa menggunakan pustaka eksternal seperti `sklearn`.
     - **Fungsi `fit`**: Melakukan pelatihan model dengan menggunakan rumus regresi linear.
     - **Fungsi `predict`**: Menghitung prediksi berdasarkan bobot yang telah dihitung selama pelatihan.

### **4. Evaluasi Model:**
   - **Metrik Evaluasi**:
     - Menambahkan fungsi **`mean_absolute_error (MAE)`**, **`mean_squared_error (MSE)`**, dan **`r2_score`** untuk mengevaluasi kualitas model berdasarkan data pelatihan.

### **5. Standardisasi Data:**
   - **Normalisasi Fitur**:
     - **Standardisasi** fitur dengan menghitung rata-rata dan deviasi standar untuk fitur pelatihan dan pengujian (menggunakan `X_train_mean` dan `X_train_std`).

### **6. Prediksi Data Uji:**
   - **Prediksi dengan Model**:
     - Dataset uji diproses dan dinormalisasi, lalu digunakan untuk melakukan prediksi menggunakan model yang telah dilatih.
     - Hasil prediksi `calories` disimpan dalam dataset uji dan disimpan dalam file CSV terpisah.

### **7. Penyimpanan dan Hasil:**
   - **Menyimpan Prediksi**: Hasil prediksi `calories` disimpan dalam file CSV terpisah (`Food_Dataset_test_with_calories.csv`).
   - **Menampilkan Hasil**: Menampilkan lima baris pertama dari dataset uji yang telah diprediksi di terminal.

### **Perubahan Signifikan:**
- **Normalisasi** data dilakukan untuk memastikan stabilitas pelatihan dan prediksi.
- **Fitur tambahan** untuk menangkap interaksi dan kuadrat dari data, yang membantu model untuk mempelajari hubungan yang lebih kompleks.
- **Pemrosesan nilai `calories`** dengan menghapus pemisah ribuan agar nilai tidak salah dibaca.
