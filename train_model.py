import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils

# Parameter
K = 3  # Jumlah tetangga terdekat
IMG_SIZE = 32  # Ukuran gambar yang akan digunakan (64x64)

def load_dataset(base_path):
    X = []  # Fitur
    y = []  # Label
    
    class_mapping = {
        "cardboard": 0,
        "glass": 1,
        "metal": 2,
        "paper": 3,
        "plastic": 4,
        "trash": 5
    }
    
    print("\nMemeriksa dataset:")
    for class_name, class_idx in class_mapping.items():
        folder_path = os.path.join(base_path, class_name)
        if not os.path.exists(folder_path):
            print(f"[ERROR] Folder tidak ditemukan: {folder_path}")
            continue
            
        image_count = 0
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                # Baca gambar
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARNING] Gagal membaca gambar: {img_path}")
                    continue
                    
                # Konversi ke RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize gambar
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                
                # Konversi ke grayscale
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
                
                # Normalisasi
                img_normalized = img_gray.astype(np.float32) / 255.0
                
                # Flatten gambar
                img_flattened = img_normalized.flatten()
                
                X.append(img_flattened)
                y.append(class_idx)
                image_count += 1
                
            except Exception as e:
                print(f"[ERROR] Error processing {img_path}: {e}")
                continue
        
        print(f"Kelas {class_name}: {image_count} gambar berhasil dimuat (termasuk augmentasi)")
    
    if not X:
        raise ValueError("Tidak ada gambar yang berhasil dimuat!")
        
    return np.array(X), np.array(y)

def create_visualizations(X, y, y_pred, y_test, class_names):
    """Fungsi untuk membuat visualisasi"""
    # 1. Distribusi Kelas (Plotly)
    class_counts = {class_names[i]: np.sum(y == i) for i in sorted(class_names.keys())}
    fig_dist = px.bar(
        x=list(class_counts.keys()),
        y=list(class_counts.values()),
        labels={'x': 'Kategori', 'y': 'Jumlah Sampel'},
        title='Distribusi Jumlah Sampel per Kelas',
        color_discrete_sequence=['#17949b']
    )
    distribution_plot_json = json.dumps(fig_dist, cls=plotly.utils.PlotlyJSONEncoder)

    # 2. Confusion Matrix (Plotly)
    cm = confusion_matrix(y_test, y_pred)
    
    # Buat label untuk sumbu X dan Y
    cm_labels = list(class_names.values())
    
    fig_cm = go.Figure(data=go.Heatmap(
                       z=cm,
                       x=cm_labels,
                       y=cm_labels,
                       colorscale='Blues',
                       showscale=False))

    fig_cm.update_layout(
        title='Confusion Matrix',
        xaxis_title='Prediksi',
        yaxis_title='Aktual',
        xaxis_nticks=len(cm_labels),
        yaxis_nticks=len(cm_labels),
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        margin=dict(l=30, r=30, b=80, t=80, pad=4) # Sesuaikan margin
    )

    # Tambahkan angka ke heatmap
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            fig_cm.add_annotation(
                x=cm_labels[j],
                y=cm_labels[i],
                text=str(cm[i, j]),
                showarrow=False,
                font=dict(color="black", size=12)
            )

    confusion_matrix_graph_json = json.dumps(fig_cm, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 3. Contoh Gambar per Kelas (Matplotlib - tetap sebagai base64)
    plt.figure(figsize=(15, 10))
    for i, class_name in class_names.items():
        # Ambil 5 gambar pertama dari setiap kelas
        class_indices = np.where(y == i)[0][:5]
        for j, idx in enumerate(class_indices):
            plt.subplot(len(class_names), 5, i*5 + j + 1)
            img = X[idx].reshape(IMG_SIZE, IMG_SIZE)
            plt.imshow(img, cmap='gray')
            plt.title(f'{class_name}\nSampel {j+1}')
            plt.axis('off')
    plt.tight_layout()
    
    # Simpan contoh gambar
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    sample_images = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return {
        'distribution_plot': distribution_plot_json,
        'confusion_matrix_graph': confusion_matrix_graph_json,
        'sample_images': sample_images
    }

def main():
    # Definisikan class_names di awal
    class_names = {
        0: "Karton",
        1: "Kaca",
        2: "Kaleng",
        3: "Kertas",
        4: "Plastik",
        5: "Sampah Lainnya"
    }
    
    # Load dataset
    print("Loading dataset...")
    X, y = load_dataset('dataset/Garbage classification')
    
    # Normalisasi data
    print("\nNormalizing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data (80% training, 20% testing)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Print jumlah sampel per kelas
    print("\nJumlah sampel per kelas:")
    for label in np.unique(y):
        count = np.sum(y == label)
        print(f"{class_names[label]}: {count} sampel")
    
    # Train KNN model
    print("\nTraining KNN model...")
    model = KNeighborsClassifier(n_neighbors=K, weights='distance', metric='manhattan')
    model.fit(X_train, y_train)
    
    # Evaluasi model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Generate classification report dengan zero_division=0
    report = classification_report(y_test, y_pred, 
                                 target_names=list(class_names.values()),
                                 zero_division=0)
    print("\nClassification Report:")
    print(report)
    
    # Buat visualisasi
    print("\nMembuat visualisasi...")
    visualizations = create_visualizations(X, y, y_pred, y_test, class_names)
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save analysis results
    analysis_results = {
        'accuracy': accuracy,
        'classification_report': report,
        'distribution_plot': visualizations['distribution_plot'],
        'confusion_matrix_graph': visualizations['confusion_matrix_graph'],
        'sample_images': visualizations['sample_images']
    }
    
    with open('analysis_results.json', 'w') as f:
        json.dump(analysis_results, f)
    
    print("Training completed!")

if __name__ == "__main__":
    main()