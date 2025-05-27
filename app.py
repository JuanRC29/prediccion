from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Cargar el modelo entrenado
with open("modelo_recomendacion.pkl", "rb") as f:
    model = pickle.load(f)

# Columnas exactas que usaste en X
features = [
    'Descuento_Ofertado', 'Num_Productos', 'Frecuencia_Cliente', 'Ticket_Promedio',
    'carrito_Agua', 'carrito_Aguacate', 'carrito_Alimento para Gatos', 'carrito_Alimento para Perros',
    'carrito_Arena para Gatos', 'carrito_Banano', 'carrito_Barra de Cereal', 'carrito_Cerveza',
    'carrito_Chocolate', 'carrito_Cloro', 'carrito_Detergente', 'carrito_Galletas',
    'carrito_Gaseosa', 'carrito_Jabón', 'carrito_Jugo', 'carrito_Leche',
    'carrito_Lechuga', 'carrito_Mantequilla', 'carrito_Manzana', 'carrito_Papas Fritas',
    'carrito_Queso', 'carrito_Tomate', 'carrito_Yogur'
]

app = Flask(__name__)

@app.route('/predecir', methods=['POST'])
def predecir():
    data = request.json
    df = pd.DataFrame([data], columns=features)
    prob = model.predict_proba(df)[0][1]
    return jsonify({"probabilidad": round(prob, 4)})

if __name__ == '__main__':
    app.run(debug=True)
