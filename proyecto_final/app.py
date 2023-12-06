from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import os
from clases import DataCleaning, PricePredictionModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'archivos'

# Inicializar variables globales
model = None
cleaned_data = None

@app.route('/')
def upload():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file_post():
    global cleaned_data
    # Crear la carpeta si no existe
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No file selected')
    if file.filename.split('.')[-1] != 'csv':
        return render_template('index.html', message='The file must be a CSV file')
    else:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            dataset = pd.read_csv(file_path)
            data_cleaner = DataCleaning(dataset)
            cleaned_data = data_cleaner.purify_data()
            return render_template('index.html', message='File successfully uploaded and cleaned')
        except Exception as e:
            print(e)  # Esto imprimirá la excepción en la consola
            return render_template('index.html', message=f'An error occurred: {e}')

@app.route('/feed')
def feed_the_model():
    global model, cleaned_data
    if cleaned_data is not None:
        try:
            model = PricePredictionModel(cleaned_data)
            model.execute_regression_analysis()
            return render_template('index.html', message='Model has been trained')
        except Exception as e:
            print(e)
            return render_template('index.html', message=f'An error occurred while training the model: {e}')
    else:
        return render_template('index.html', message='No data available for training')

@app.route('/predict', methods=['POST'])
def prediction():
    global model
    if model:
        try:
            # Recuperar datos de entrada del usuario
            horsepower = float(request.form.get('horsepower', 0))
            citympg = float(request.form.get('citympg', 0))
            highwaympg = float(request.form.get('highwaympg', 0))
            enginesize = float(request.form.get('enginesize', 0))
            gas = float(request.form.get('gas', 0))

            # Crear DataFrame para predicción
            user_input = pd.DataFrame({
                'horsepower': [horsepower],
                'citympg': [citympg],
                'highwaympg': [highwaympg],
                'enginesize': [enginesize],
                'gas': [gas]
            })

            # Realizar predicción
            user_prediction = model.make_user_prediction(user_input)

            # Guardar la predicción en el archivo resultados.txt
            with open(os.path.join(app.config['UPLOAD_FOLDER'], 'resultados.txt'), 'w') as f:
                f.write(f'Predicted price: {user_prediction[0]:,.2f}')

            return render_template('index.html', message='Prediction saved to resultados.txt')
        except Exception as e:
            print(e)
            return render_template('index.html', message=f'An error occurred while making the prediction: {e}')
    else:
        return render_template('index.html', message='Model is not trained')

@app.route('/download')
def descarga():
    try:
        # Asegúrate de que el archivo 'resultados.txt' exista antes de intentar descargarlo
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'resultados.txt')):
            return send_from_directory(app.config['UPLOAD_FOLDER'], 'resultados.txt', as_attachment=True)
        else:
            return render_template('index.html', message='Output file not found')
    except Exception as e:
        print(e)
        return render_template('index.html', message=f'An error occurred while downloading the file: {e}')

if __name__ == "__main__":
    app.run(debug=True)
