import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Tentukan direktori dataset lokal
import os

# Membaca lokasi absolut dari file cnn_rps.py secara otomatis
current_dir = os.path.dirname(os.path.abspath(__file__))

# Mengunci posisi folder dataset tepat di samping file script ini
base_dir = os.path.join(current_dir, 'rockpaperscissors')

# Proteksi jika folder dataset belum diekstrak atau salah penamaan
if not os.path.exists(base_dir):
    print(f"Error: Folder dataset '{base_dir}' tidak ditemukan!")
    print("Pastikan kamu sudah mengekstrak dataset dan menaruhnya di folder yang sama dengan script ini.")
    exit()

# 2. Augmentasi Data & Pemisahan Dataset (Train & Validation Split)
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2  # 20% untuk data validasi, 80% untuk data latih
)

# Generator untuk Data Training (80%)
print("--- Memuat Data Training ---")
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Multi-kelas (rock, paper, scissors)
    subset='training'          # Mengambil bagian training
)

# Generator untuk Data Validation (20%)
print("\n--- Memuat Data Validation ---")
validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'        # Mengambil bagian validation
)

# Menampilkan indeks kelas yang terdeteksi secara otomatis oleh folder
print(f"\nIndeks Kelas: {train_generator.class_indices}")

# 3. Membangun Arsitektur Model CNN
model = Sequential([
    # Convolutional Layer 1 & Max Pooling 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    
    # Convolutional Layer 2 & Max Pooling 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Convolutional Layer 3 & Max Pooling 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Flatten Layer (Mengubah matriks 2D menjadi vektor 1D)
    Flatten(),
    
    # Fully Connected Layer (Hidden & Output)
    Dense(512, activation='relu'),
    Dense(3, activation='softmax') 
])

# 4. Menampilkan Ringkasan Arsitektur Model
model.summary()

# 5. Kompilasi Model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 6. Pelatihan Model (Model Fitting)
print("\n=== Memulai Pelatihan Model (CNN) ===")
history = model.fit(
    train_generator,
    epochs=10,  
    validation_data=validation_generator
)

# 7. Evaluasi Model pada Data Validasi
print("\n=== Evaluasi Model ===")
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation Loss     : {val_loss:.4f}")
print(f"Validation Accuracy : {val_acc:.4f}")

# 8. Visualisasi Metrik Akurasi & Loss Selama Pelatihan
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.title('Grafik Performa Pelatihan CNN (Rock-Paper-Scissors)')
plt.xlabel('Epochs')
plt.ylabel('Nilai')
plt.grid(True)
plt.show()