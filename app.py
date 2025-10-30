# app.py
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model dan encoder
try:
    model = joblib.load('model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
except FileNotFoundError as e:
    raise RuntimeError(f"File model atau encoder tidak ditemukan: {e}")

# Daftar fitur sesuai urutan dataset (tanpa Loan_ID dan Loan_Status)
FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        input_data = []

        for feature in FEATURES:
            if feature not in data:
                return jsonify({'error': f"Input '{feature}' wajib diisi."}), 400

            value = data[feature].strip()
            if not value:
                return jsonify({'error': f"Nilai untuk '{feature}' tidak boleh kosong."}), 400

            # Encode fitur kategorikal
            if feature in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
                le = label_encoders[feature]
                # Handle unseen label (sebaiknya dihindari, tapi antisipasi)
                if value not in le.classes_:
                    # Tambahkan label baru sementara (hanya untuk demo)
                    le.classes_ = list(le.classes_) + [value]
                encoded_val = le.transform([value])[0]
                input_data.append(encoded_val)
            else:
                # Fitur numerik
                try:
                    input_data.append(float(value))
                except ValueError:
                    return jsonify({'error': f"Nilai '{value}' pada '{feature}' bukan angka yang valid."}), 400

        # Prediksi probabilitas
        proba = model.predict_proba([input_data])[0]
        prob_accept = proba[1]  # Probabilitas "Disetujui"

        # Gunakan threshold 60% untuk keputusan bisnis
        if prob_accept >= 0.6:
            result = "✅ Disetujui"
        else:
            result = "❌ Ditolak"

        confidence = f"{prob_accept * 100:.1f}%"

        return jsonify({
            'prediction': result,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': f"Terjadi kesalahan: {str(e)}"}), 500

# Untuk Railway
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)