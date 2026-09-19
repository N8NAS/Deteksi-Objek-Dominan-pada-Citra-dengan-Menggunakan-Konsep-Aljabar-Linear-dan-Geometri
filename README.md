# Deteksi Objek Dominan pada Citra dengan Konsep Aljabar Linear dan Geometri

Repositori ini berisi implementasi algoritma untuk mendeteksi objek dominan pada sebuah citra (gambar) menggunakan konsep Aljabar Linear, secara khusus memanfaatkan **Singular Value Decomposition (SVD)** dan ekstraksi fitur. Proyek ini merupakan bagian dari tugas mata kuliah Aljabar Linear dan Geometri (Algeo).

## Struktur Repositori

- `src/` : Berisi source code utama program Python (`deteksiObjekUtamaAlgeo.py`).
- `docs/` : Berisi dokumen makalah/laporan yang menjelaskan teori dan implementasi dari metode yang digunakan.

## Prasyarat (Prerequisites)

Untuk menjalankan kode pada repositori ini, pastikan Anda telah menginstal library Python berikut:

- `numpy`
- `Pillow` (PIL)

Anda dapat menginstalnya menggunakan pip:
```bash
pip install numpy Pillow
```

## Cara Penggunaan

1. Siapkan gambar yang ingin Anda proses (misalnya `image.jpg`).
2. Buka file `src/deteksiObjekUtamaAlgeo.py`.
3. **Penting:** Secara default, program akan membaca gambar bernama `image.jpg`. Untuk melakukan testing dengan gambar pribadi, silakan ganti path gambar yang ada di sekitar **baris 41** dengan path gambar Anda:
   ```python
   img = Image.open("path/to/gambar_anda.jpg").convert("L")
   ```
4. Jalankan script Python tersebut:
   ```bash
   cd src
   python deteksiObjekUtamaAlgeo.py
   ```
5. Program akan menghasilkan dua buah file output di direktori yang sama:
   - `objek dominan.png` : Gambar yang berisi objek utama hasil deteksi.
   - `objek lain.png` : Gambar yang berisi latar belakang atau objek selain objek utama.

## Konsep yang Digunakan

Program ini bekerja dengan cara:
1. Mengubah gambar menjadi grayscale (hitam putih) dan merepresentasikannya sebagai matriks.
2. Melakukan proses *centering* pada matriks gambar.
3. Mengaplikasikan *Singular Value Decomposition* (SVD) untuk mengekstrak *k* fitur utama (secara default k=8).
4. Menghitung jarak (*Euclidean distance*) dari fitur setiap piksel terhadap *centroid* fitur keseluruhan.
5. Memisahkan objek dominan dan *background* berdasarkan nilai *threshold* (rata-rata jarak).
