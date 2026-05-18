import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Menghilangkan Info dan Warning logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Menghilangkan oneDNN logs

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def main():
    print("=== Praktikum KB Pertemuan 7: Jaringan Syaraf Tiruan 2 ===")
    
    # 2. Muat dataset iris dari file CSV/data lokal
    print("\n[INFO] Memuat dataset Iris dari file lokal...")
    dataset = pd.read_csv('iris/iris.data', header=None, sep=',')

    # Menyusun data X (fitur) dan y (label)
    X = dataset.iloc[:, :-1].values # 4 kolom pertama sebagai fitur
    y = dataset.iloc[:, -1].values # Kolom terakhir sebagai label

    # 3. Mengonversi label dari string menjadi numerik
    print("[INFO] Mengonversi label...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y) # Mengubah label jadi 0, 1, 2

    # 4. Memisahkan dataset menjadi data latih dan data validasi dengan rasio 80:20
    print("[INFO] Membagi dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Buat model neural network dengan 1 layer input dan 4 layer Dense
    print("\n[INFO] Membangun Arsitektur Model JST...")
    model = Sequential([
        Input(shape=X_train.shape[1:]),
        Dense(1000, activation='relu'),
        Dense(500, activation='relu'),
        Dense(300, activation='relu'),
        Dense(3, activation='softmax')
    ])

    # Tampilkan summary arsitektur model
    model.summary()

    # 6. Kompilasi model
    print("\n[INFO] Mengkompilasi model...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 7. Model dilatih menggunakan dataset training
    print("\n[INFO] Memulai pelatihan model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # 8. Evaluasi model pada data validasi
    print("\n[INFO] Mengevaluasi model pada data testing...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    # 9. Visualisasikan perubahan loss dan accuracy
    print("\n[INFO] Menampilkan grafik loss dan accuracy (Tutup window grafik untuk melanjutkan)...")
    pd.DataFrame(history.history).plot(figsize=(10,6))
    plt.title('Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.show()

    # 10. Lakukan prediksi pada data baru
    print("\n[INFO] Melakukan prediksi pada data testing...")
    predictions = model.predict(X_test)
    predicted_classes = predictions.argmax(axis=1)

    print("Prediksi  :", predicted_classes)
    print("Label Asli:", y_test)

    # 11. Buat confusion matrix
    print("\n[INFO] Menampilkan Confusion Matrix (Tutup window grafik untuk melanjutkan)...")
    cm = confusion_matrix(y_test, predicted_classes)

    # Visualisasikan confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # 12. Fungsi interaktif untuk prediksi input baru
    print("\n=== Prediksi Data Baru ===")
    predict_new_data(model, label_encoder)

def predict_new_data(model, label_encoder):
    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width = float(input("Masukkan sepal width: "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width = float(input("Masukkan petal width: "))
    
    # Membuat data array baru
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Melakukan prediksi
    prediction = model.predict(new_data)
    predicted_class = prediction.argmax(axis=1)
    
    # Mengonversi hasil prediksi numerik menjadi label asli
    predicted_label = label_encoder.inverse_transform(predicted_class)
    print(f"Prediksi kelas: {predicted_label[0]}")

if __name__ == '__main__':
    main()
