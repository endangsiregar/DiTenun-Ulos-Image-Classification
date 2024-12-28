from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import joblib
import os
import logging
from werkzeug.utils import secure_filename
import tempfile

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Direktori sementara untuk menyimpan file yang diunggah
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Inisialisasi logging
logging.basicConfig(level=logging.INFO)

# Load model dan encoder
ulos_knn_model = joblib.load("ulos_knn_model.pkl")

# Informasi terkait ulos
ulos_details = {
    "Ulos Ragihotang": {
        "manfaat": "Melambangkan kesuburan dan kekuatan.",
        "tujuan": "Digunakan dalam acara pernikahan adat.",
        "deskripsi": "Ulos ini memiliki pola yang khas dengan warna dominan merah dan hitam."
    },
    "Ulos Sibolang": {
        "manfaat": "Simbol keprihatinan atau duka.",
        "tujuan": "Digunakan dalam upacara adat terkait kematian.",
        "deskripsi": "Memiliki pola dengan warna biru tua dan hitam sebagai lambang keteguhan."
    },
    "Ulos Sadum": {
        "manfaat": "Melambangkan harapan dan doa.",
        "tujuan": "Sering diberikan sebagai hadiah untuk pasangan baru menikah.",
        "deskripsi": "Dikenal dengan pola garis warna-warni yang ceria."
    },
    "Ulos Ragiidup": {
        "manfaat": "Simbol kesejahteraan dan kehidupan.",
        "tujuan": "Dipakai dalam acara adat untuk memberkati seseorang.",
        "deskripsi": "Warna merah dominan dengan pola geometris unik."
    },
    "Ulos Bintang Maratur": {
        "manfaat": "Simbol keharmonisan dan keberuntungan.",
        "tujuan": "Digunakan dalam acara adat yang mengharapkan keseimbangan.",
        "deskripsi": "Memiliki pola bintang berulang dengan warna cerah."
    }
}

# Mapping label ke nama ulos
label_mapping = {
    0: "Ulos Ragihotang",
    1: "Ulos Sibolang",
    2: "Ulos Sadum",
    3: "Ulos Ragiidup",
    4: "Ulos Bintang Maratur"
}

# Validasi ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi untuk memproses gambar
def preprocess_image(image_path, target_size=(64, 64)):
    with Image.open(image_path) as img:
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized).astype('float32') / 255.0  # Normalisasi
        img_flattened = img_array.flatten()  # Ubah menjadi array 1D
    return img_flattened.reshape(1, -1)

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template("index.html")

# Route untuk prediksi ulos
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template("index.html", error="Tidak ada gambar yang diunggah.")

    file = request.files['image']

    if file.filename == '' or not allowed_file(file.filename):
        return render_template("index.html", error="Format file tidak valid. Hanya mendukung PNG, JPG, dan JPEG.")

    try:
        # Simpan file gambar
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_GAMBAR'], filename)
        file.save(image_path)

        # Preprocessing gambar
        processed_image = preprocess_image(image_path)

        # Prediksi dengan model
        prediction = ulos_knn_model.predict(processed_image)[0]
        ulos_name = label_mapping.get(prediction)

        if ulos_name is None or ulos_name not in ulos_details:
            return render_template(
                "result.html",
                ulos_name="Tidak Dikenali",
                manfaat="Gambar yang diunggah bukan gambar ulos yang valid.",
                tujuan="Silakan unggah gambar ulos yang sesuai.",
                deskripsi="Model tidak dapat mengidentifikasi gambar ini sebagai salah satu ulos.",
                image_path=image_path
            )

        ulos_info = ulos_details[ulos_name]

        return render_template(
            "result.html",
            ulos_name=ulos_name,
            manfaat=ulos_info["manfaat"],
            tujuan=ulos_info["tujuan"],
            deskripsi=ulos_info["deskripsi"],
            image_path=image_path
        )
    except Exception as e:
        logging.error(f"Kesalahan: {str(e)}")
        return render_template("index.html", error=f"Kesalahan terjadi: {str(e)}")

# Menjalankan aplikasi
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8000)
