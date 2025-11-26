#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#


import os
import json
import gzip
import pickle
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error


df_train = pd.read_csv("files/input/train_data.csv.zip", compression="zip").copy()
df_test = pd.read_csv("files/input/test_data.csv.zip", compression="zip").copy()

df_train["Age"] = 2021 - df_train["Year"]
df_test["Age"] = 2021 - df_test["Year"]

df_train = df_train.drop(columns=["Year", "Car_Name"])
df_test = df_test.drop(columns=["Year", "Car_Name"])
X_tr = df_train.drop(columns=["Present_Price"])
y_tr = df_train["Present_Price"]

X_te = df_test.drop(columns=["Present_Price"])
y_te = df_test["Present_Price"]

categoricas = ["Fuel_Type", "Selling_type", "Transmission"]
numericas = [col for col in X_tr.columns if col not in categoricas]

preproc = ColumnTransformer(
    transformers=[
        ("cat_cols", OneHotEncoder(), categoricas),
        ("num_cols", MinMaxScaler(), numericas),
    ]
)

modelo_pipe = Pipeline(
    steps=[
        ("preprocess", preproc),
        ("kbest", SelectKBest(score_func=f_regression)),
        ("regressor", LinearRegression()),
    ]
)
hiperparametros = {
    "kbest__k": range(1, 15),
    "regressor__fit_intercept": [True, False],
    "regressor__positive": [True, False],
}

busqueda = GridSearchCV(
    estimator=modelo_pipe,
    param_grid=hiperparametros,
    cv=10,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    refit=True,
)

busqueda.fit(X_tr, y_tr)
os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(busqueda, f)
pred_tr = busqueda.predict(X_tr)
pred_te = busqueda.predict(X_te)

metricas_train = {
    "type": "metrics",
    "dataset": "train",
    "r2": float(r2_score(y_tr, pred_tr)),
    "mse": float(mean_squared_error(y_tr, pred_tr)),
    "mad": float(median_absolute_error(y_tr, pred_tr)),
}

metricas_test = {
    "type": "metrics",
    "dataset": "test",
    "r2": float(r2_score(y_te, pred_te)),
    "mse": float(mean_squared_error(y_te, pred_te)),
    "mad": float(median_absolute_error(y_te, pred_te)),
}
Path("files/output").mkdir(parents=True, exist_ok=True)

with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(metricas_train) + "\n")
    f.write(json.dumps(metricas_test) + "\n")