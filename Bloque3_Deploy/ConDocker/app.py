# app.py - Ejemplo de API con Flask

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo
model = joblib.load('model.pkl')

# Opcional: Cargar scaler u otros preprocesadores
# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

@app.route('/')
def home():
    return jsonify({
        'message': 'API de Machine Learning',
        'version': '1.0',
        'endpoints': ['/predict']
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        # Realizar predicción
        prediction = model.predict(features)
        
        return jsonify({
            'prediction': prediction.tolist(),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Render asigna el puerto dinámicamente
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)