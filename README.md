> **Chapter 15: Deteksi Objek & Pengenalan Karakter Berbasis Visi Komputer**

---

## 👤 Identitas Mahasiswa
* **Nama** : Tiara
* **Studi** : Semester 4 — Teknik Informatika / Sistem Informasi
* **Matkul**: Pengolahan Citra Digital

---

## 🎯 Daftar Proyek & Capaian
| Tugas | Deskripsi Proyek | Metode / Teori Utama | Status |
| :--- | :--- | :--- | :---: |
| **01. Summary** | Analisis & ringkasan paper deteksi pedestrian | Histograms of Oriented Gradients (HOG) | ✨ Selesai |
| **02. Modul** | Deteksi & *Object Tracking* Manusia pada Video | HOG Descriptor + Centroid Tracker | ✨ Selesai |
| **03. Kolom** | *Automatic Number Plate Recognition* (ANPR) | Canny Edge + Pytesseract OCR | ✨ Selesai |

---

## 📁 Arsitektur Direktori Kerja
Susunan berkas diatur secara linear agar memudahkan eksekusi modul tanpa memicu kendala *pathing*:


```

```text
Aesthetic README.md successfully created.

```🗂️
Chapter15_Citra_Digital/
├── 📄 README.md               <- Dokumen laporan utama (Aesthetic Report)
├── 📜 Dalal-cvpr05.pdf        <- Dokumen referensi akademik Tugas 1
│
├── 📂 02_Pedestrian_Detector/
│   ├── 🐍 deteksi_gambar.py   <- Ekstraksi koordinat pejalan kaki statis
│   ├── 🐍 deteksi_video.py    <- Real-time tracking & penguncian ID objek
│   ├── 🖼️ img.png             <- Aset pengujian citra statis
│   └── 🎥 vid.mp4             <- Aset pengujian citra dinamis
│
└── 📂 03_Plate_Recognition/
    ├── 🐍 deteksi_plat.py     <- Pipeline segmentasi & OCR plat nomor
    └── 🖼️ plat_nomor.jpg      <- Aset pengujian plat nomor kendaraan

```

---

## 📝 1. Tugas Summary Paper Akademik

### 📄 Informasi Jurnal

* **Judul**: *Histograms of Oriented Gradients for Human Detection*
* **Penulis**: Navneet Dalal & Bill Triggs
* **Afiliasi**: INRIA Rhône-Alps, Prancis (CVPR 2005)

### 🔬 Intisari & Metodologi Utama

Algoritma **HOG (Histogram of Oriented Gradients)** bekerja dengan prinsip mendasar bahwa bentuk dan kemunculan objek lokal di dalam gambar dapat dikarakterisasi dengan sangat baik melalui distribusi gradien intensitas tepi cahaya.

```⚙️ Alur Pemrosesan HOG:
Gambar Masukan ──> Gradien Filter ──> Binning Orientasi (9 Saluran) ──> Normalisasi Blok Kontras ──> Linear SVM Classifier

```

1. **Komputasi Gradien**: Citra diproses menggunakan masker linear sederhana `[-1, 0, 1]` guna mendeteksi kontur tajam tanpa pelembutan gambar (*smoothing*) berlebih.
2. **Spatial Binning (*Cells*)**: Gambar dibagi ke dalam area spasial kecil bernama *Cells* ($6\times6$ hingga $8\times8$ piksel). Setiap piksel memberikan hak suara (*vote*) berdasarkan arah sudut gradien ($0^\circ - 180^\circ$).
3. **Normalisasi Kontras (*Blocks*)**: Beberapa sel dikelompokkan ke dalam struktur yang lebih besar bernama *Blocks* yang saling tumpang tindih (*overlapping*). Kontras dinormalisasi secara lokal pada tingkat blok untuk memberikan ketahanan terhadap bayangan ekstrem.
4. **Klasifikasi**: Gabungan vektor fitur dari seluruh blok dimasukkan ke dalam **Linear Support Vector Machine (SVM)** untuk memisahkan objek manusia dari latar belakang kompleks.

---

## 🏃‍♂️ 2. Tugas Modul: Deteksi & Pelacakan Pejalan Kaki

### 🛠️ Fitur Unggulan & Optimasi

Program bawaan modul asli hanya mampu melakukan **Deteksi Standar** (kotak berkedip di setiap frame tanpa memori). Kode ini telah **ditingkatkan secara signifikan** dengan menyematkan **Centroid Tracker**:

* 🔹 **Penguncian Target**: Mengunci koordinat pejalan kaki dari frame ke frame.
* 🔹 **Pemberian ID Unik**: Menempelkan teks dinamis (`Orang ID: 0`, `Orang ID: 1`) yang bergerak mengikuti pergeseran objek secara kontinu.

### 💻 Implementasi Kode Program (`deteksi_video.py`)

```python
# ==============================================================================
# PROYEK DETEKSI DAN PELACAKAN MANUSIA - REALTIME TRACKER
# ==============================================================================
import cv2
import imutils
import numpy as np

# Inisialisasi Detektor HOG bawaan OpenCV yang sudah terlatih (Pre-trained)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Membaca stream berkas video masukan
cap = cv2.VideoCapture('vid.mp4')

next_object_id = 0
tracked_objects = {}  # Struktur data memori pelacak: {ID: (x, y, w, h)}

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
        
    image = imutils.resize(image, width=min(400, image.shape[1]))
    
    # Deteksi multi-skala objek manusia
    (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
    
    current_centroids = []
    current_rects = []
    
    for (x, y, w, h) in regions:
        cX = int(x + w / 2.0)
        cY = int(y + h / 2.0)
        current_centroids.append((cX, cY))
        current_rects.append((x, y, w, h))
        
    # --- PIPELINE CENTROID TRACKING ---
    new_tracked_objects = {}
    for i, (cX, cY) in enumerate(current_centroids):
        matched_id = None
        min_distance = 50  # Batas toleransi jarak piksel pergeseran objek
        
        for obj_id, (ox, oy, ow, oh) in tracked_objects.items():
            ocX = int(ox + ow / 2.0)
            ocY = int(oy + oh / 2.0)
            distance = np.sqrt((cX - ocX)**2 + (cY - ocY)**2)
            
            if distance < min_distance:
                matched_id = obj_id
                min_distance = distance
                
        if matched_id is not None:
            new_tracked_objects[matched_id] = current_rects[i]
        else:
            new_tracked_objects[next_object_id] = current_rects[i]
            next_object_id += 1
            
    tracked_objects = new_tracked_objects
    
    # --- RENDER VISUALISASI ---
    for obj_id, (x, y, w, h) in tracked_objects.items():
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, f"Orang ID: {obj_id}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Pedestrian Tracking System", image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```

---

## 🚗 3. Tugas Kolom Tugas: Pengenalan Plat Nomor (ANPR)

### ⚙️ Pipeline Arsitektur Pengolahan Citra

Sistem ANPR (*Automatic Number Plate Recognition*) ini dirancang melalui 6 tahapan pemrosesan digital:

```📊
[Gambar Mobil] ──> (Grayscale) ──> (Gaussian Blur) ──> (Canny Edge) ──> [Deteksi Kontur Poligon] ──> (Threshold Biner) ──> [Tesseract OCR]

```

* **Grayscale & Smoothing**: Mengurangi beban komputasi matriks RGB dan membuang derau (*noise*) halus pada permukaan bodi mobil menggunakan filter Gaussian Blur.
* **Canny Edge Detection**: Mengekstrak diskontinuitas intensitas cahaya untuk memunculkan bentuk tepi struktural dari plat nomor kendaraan.
* **Kontur Poligon (`approxPolyDP`)**: Memindai objek berbentuk persegi panjang tertutup yang memiliki tepat 4 simpul sudut sebagai lokasi mutlak plat nomor.
* **OTSU Binarization**: Mengubah area potongan plat nomor menjadi hitam-putih tegas guna memisahkan piksel teks karakter dengan latar dasar plat nomor.

### 📝 Catatan Penting Pengujian & Validasi Lingkungan

1. **Instalasi Eksternal**: Program ini membutuhkan mesin utama Tesseract OCR terpasang pada Windows via perintah:
```bash
winget install UB-Mannheim.TesseractOCR

```


2. **Evaluasi Deteksi Kontur**: Berdasarkan uji coba, visualisasi *Canny Edge* telah berhasil membentuk pola kontur persegi panjang yang presisi pada area plat nomor mobil.
3. **Optimasi Akurasi OCR**: Untuk menghasilkan pembacaan karakter string yang 100% akurat tanpa distorsi huruf, disarankan menggunakan gambar masukan dengan sudut pemotretan yang tegak lurus (*frontal shot*) serta kualitas pencahayaan yang merata.

---

💡 *Laporan Praktikum Selesai Disusun — Jun 2026.*
