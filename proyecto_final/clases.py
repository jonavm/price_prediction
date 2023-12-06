import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os

class DataCleaning:
    def __init__(self, dataframe):
        self.df = dataframe

    def purify_data(self):
        # Eliminar duplicados
        self.df.drop_duplicates(inplace=True)

        # Eliminar filas con valores NaN en columnas clave
        self.df.dropna(subset=['price', 'horsepower', 'citympg', 'highwaympg', 'enginesize'], inplace=True)

        # Convertir 'fueltype' a variables dummy
        fueltype_dummies = pd.get_dummies(self.df['fueltype'], drop_first=True)
        self.df = pd.concat([self.df, fueltype_dummies], axis=1)

        # Eliminar columnas innecesarias
        self.df.drop(columns=['car_ID', 'CarName', 'fueltype'], inplace=True)

        return self.df

class PricePredictionModel:
    def __init__(self, data):
        self.data = data

    def execute_regression_analysis(self):
        # Preparar los datos para el modelo
        X = self.data[['horsepower', 'citympg', 'highwaympg', 'enginesize', 'gas']]
        y = self.data['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar el modelo
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Evaluar el modelo
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        r_squared = self.model.score(X_test, y_test)

        # Imprimir métricas
        print('Mean Absolute Error:', mae)
        print('Mean Squared Error:', mse)
        print('Root Mean Squared Error:', rmse)
        print('R^2:', r_squared)

    def make_user_prediction(self, user_input):
        # Realizar predicción basada en la entrada del usuario
        user_prediction = self.model.predict(user_input)
        return user_prediction

    def store_outcomes(self, filename='resultados.txt'):
        # Crear la carpeta 'archivos' si no existe
        if not os.path.exists('archivos'):
            os.makedirs('archivos')

        # Guardar las métricas en un archivo
        with open(os.path.join('archivos', filename), 'w') as file:
            original_stdout = sys.stdout  # Guardar la referencia de la salida estándar
            sys.stdout = file  # Cambiar la salida estándar al archivo

            # Imprimir las métricas
            self.execute_regression_analysis()

            sys.stdout = original_stdout  # Restablecer la salida estándar a la consola

        return os.path.join('archivos', filename)
