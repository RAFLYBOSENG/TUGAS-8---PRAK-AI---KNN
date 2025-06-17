from flask import Flask, request, render_template
import cv2
import numpy as np
import joblib
import os
import plotly.express as px
import json
import plotly.utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Parameter
IMG_SIZE = 32  # Ukuran gambar yang akan digunakan (64x64)

# Muat model
try:
    model = joblib.load('model.pkl')
except Exception as e:
    raise FileNotFoundError("File model.pkl tidak ditemukan. Jalankan train_model.py terlebih dahulu.")

# Muat scaler
try:
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    raise FileNotFoundError("File scaler.pkl tidak ditemukan. Jalankan train_model.py terlebih dahulu.")

# Mapping hasil prediksi
class_names = {
    0: "Karton",
    1: "Kaca",
    2: "Kaleng",
    3: "Kertas",
    4: "Plastik",
    5: "Sampah Lainnya"
}

def preprocess_image(img):
    """Fungsi untuk preprocessing gambar"""
    # Resize gambar
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    # Konversi ke grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # Normalisasi
    img_normalized = img_gray.astype(np.float32) / 255.0
    
    # Flatten gambar
    img_flattened = img_normalized.flatten()
    
    return img_flattened

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Tidak ada file yang diunggah'
        
        file = request.files['file']
        if file.filename == '':
            return 'Tidak ada file yang dipilih'
        
        if file:
            try:
                # Baca gambar
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                
                # Konversi ke RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Preprocessing gambar
                img_flattened = preprocess_image(img)
                
                # Normalisasi dengan scaler yang sama
                img_scaled = scaler.transform(img_flattened.reshape(1, -1))
                
                # Prediksi menggunakan KNN
                prediction = model.predict(img_scaled)
                # Hitung probabilitas berdasarkan jarak ke K tetangga terdekat
                distances, indices = model.kneighbors(img_scaled)
                # Konversi jarak ke probabilitas (semakin dekat, semakin tinggi probabilitas)
                probabilities = 1 / (1 + distances)
                probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
                
                # Ambil kelas dengan probabilitas tertinggi
                predicted_class = class_names[prediction[0]]
                confidence = probabilities[0][0] * 100
                
                return render_template('index.html', 
                                     prediction=predicted_class,
                                     confidence=confidence,
                                     image=file.filename)
            except Exception as e:
                return f'Error: {str(e)}'
    
    return render_template('index.html')

@app.route("/analysis")
def analysis():
    # Ambil data distribusi kelas otomatis dari folder dataset
    base_path = 'dataset/Garbage classification'
    class_mapping = {
        0: "cardboard",
        1: "glass",
        2: "metal",
        3: "paper",
        4: "plastic",
        5: "trash"
    }

    class_counts = {}
    for label, folder_name in class_mapping.items():
        folder_path = os.path.join(base_path, folder_name)
        count = len(os.listdir(folder_path)) if os.path.exists(folder_path) else 0
        class_counts[class_names[label]] = count

    # Buat grafik distribusi menggunakan Plotly
    fig_dist = px.bar(
        x=list(class_counts.keys()),
        y=list(class_counts.values()),
        labels={'x': 'Kategori', 'y': 'Jumlah Gambar'},
        title='Distribusi Jumlah Gambar per Kelas',
        color_discrete_sequence=['#17949b']
    )
    graphJSON_distribution = json.dumps(fig_dist, cls=plotly.utils.PlotlyJSONEncoder)

    # Muat hasil analisis dari file JSON
    analysis_data = {}
    try:
        with open('analysis_results.json', 'r') as f:
            analysis_data = json.load(f)
    except FileNotFoundError:
        print("[WARNING] File analysis_results.json tidak ditemukan. Jalankan train_model.py terlebih dahulu.")
    except Exception as e:
        print(f"[ERROR] Gagal memuat analysis_results.json: {e}")

    # Ambil data grafik dan metrik dari data analisis
    confusion_matrix_graph = analysis_data.get('confusion_matrix_graph', None)
    sample_images = analysis_data.get('sample_images', '')
    accuracy = analysis_data.get('accuracy', None)
    classification_report_text = analysis_data.get('classification_report', '')

    return render_template(
        "analysis.html",
        distribution_graph=graphJSON_distribution,
        confusion_matrix_graph=confusion_matrix_graph,
        sample_images=sample_images,
        accuracy=accuracy,
        classification_report_text=classification_report_text
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)