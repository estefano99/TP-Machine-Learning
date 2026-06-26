# Informe - Regresion Lineal Simple

## 1. Nombre del algoritmo

Regresion lineal simple.

## 2. Objetivo del algoritmo

La regresion lineal simple es un algoritmo de Machine Learning supervisado usado para resolver problemas de regresion. Su objetivo es predecir un valor numerico continuo a partir de una sola variable de entrada.

En este proyecto, el modelo intenta predecir la progresion de la enfermedad en pacientes del dataset `Diabetes` usando solamente la variable `bmi`.

La idea principal es buscar una relacion lineal entre:

- `X`: indice de masa corporal, `bmi`;
- `y`: progresion de la enfermedad, `disease_progression`.

## 3. Dataset utilizado

Dataset: `Diabetes`.

Fuente: incluido en `scikit-learn`, mediante `sklearn.datasets.load_diabetes`.

Datos utilizados:

- Filas: 442.
- Columnas usadas en el modelo: 2 contando la variable objetivo.
- Variable independiente `X`: `bmi`.
- Variable objetivo `y`: `disease_progression`.
- Tipo de uso: una sola variable de entrada.

Se eligio este dataset porque viene incorporado en `scikit-learn`, es liviano y permite practicar una prediccion numerica sin depender de archivos externos.

## 4. Archivos modificados o creados

`main.py`

- Conecta la opcion `1. Regresion lineal simple` con la funcion `run_simple_linear_regression()`.
- Recibe la opcion ingresada por el usuario desde consola.
- No devuelve datos; coordina el menu principal.

`data/loaders.py`

- Contiene la funcion `load_diabetes_simple_regression()`.
- Carga el dataset `Diabetes`.
- Selecciona solo la columna `bmi` como variable independiente.
- Devuelve un diccionario con `x`, `y`, nombre del dataset, variable objetivo, variables usadas, cantidad de filas, cantidad de columnas y un `dataframe` para vista previa.

`models/regression.py`

- Contiene la funcion `run_simple_linear_regression()`.
- Ejecuta el flujo completo del algoritmo.
- Carga datos, separa entrenamiento y prueba, entrena el modelo, predice, calcula metricas, genera grafico y guarda informe.

`utils/dataset_preview.py`

- Contiene `print_dataset_preview()`.
- Muestra filas, columnas, variable objetivo, variables de entrada y las primeras 5 filas del dataset.

`utils/metrics.py`

- Contiene `calculate_regression_metrics()` y `print_regression_metrics()`.
- Calcula y muestra `MAE`, `MSE`, `RMSE` y `R2`.

`utils/plots.py`

- Contiene `save_simple_regression_plot()`.
- Genera un grafico con puntos reales y linea de prediccion.

`utils/reports.py`

- Contiene `save_algorithm_report()`.
- Guarda un informe `.txt` con dataset, metricas, significado de metricas, grafico e interpretacion.

## 5. Flujo de ejecucion

1. Se ejecuta `python tp_machine_learning/main.py`.
2. Se muestra el menu principal.
3. El usuario elige la opcion `1`.
4. `main.py` llama a `run_simple_linear_regression()`.
5. Se carga el dataset `Diabetes` con `load_diabetes_simple_regression()`.
6. Se separan `X` e `y`.
7. `X` contiene solo la variable `bmi`.
8. `y` contiene la progresion de la enfermedad.
9. Se separan datos de entrenamiento y prueba con `train_test_split`.
10. Se crea un modelo `LinearRegression`.
11. Se entrena el modelo con `fit(x_train, y_train)`.
12. Se generan predicciones con `predict(x_test)`.
13. Se calculan las metricas de regresion.
14. Se genera el grafico `regresion_lineal_simple_diabetes.png`.
15. Se guarda el informe `regresion_lineal_simple_diabetes.txt`.

## 6. Metricas obtenidas

Metricas reales de la ultima ejecucion:

- `MAE`: 52.2600.
- `MSE`: 4061.8259.
- `RMSE`: 63.7325.
- `R2`: 0.2334.

`MAE` mide el error absoluto promedio. Cuanto mas bajo, mejor. En este caso, el modelo se equivoca en promedio aproximadamente 52.26 unidades de progresion de enfermedad.

`MSE` mide el error cuadratico medio. Cuanto mas bajo, mejor. Penaliza mas los errores grandes porque eleva las diferencias al cuadrado.

`RMSE` es la raiz cuadrada del `MSE`. Cuanto mas bajo, mejor. Es mas facil de interpretar que el `MSE` porque queda en una escala similar a la variable objetivo.

`R2` mide que proporcion de la variacion de los datos logra explicar el modelo. Cuanto mas cerca de 1, mejor. En este caso, `0.2334` indica que usar solamente `bmi` explica una parte limitada de la progresion de la enfermedad.

## 7. Grafico generado

Archivo generado:

```text
tp_machine_learning/outputs/graphs/regresion_lineal_simple_diabetes.png
```

El grafico muestra:

- Eje X: valor de `bmi`.
- Eje Y: progresion de la enfermedad.
- Puntos azules: valores reales del conjunto de prueba.
- Linea roja: prediccion del modelo de regresion lineal.

Si los puntos estan cerca de la linea roja, significa que el modelo predijo bien esos casos. Si estan lejos, significa que hubo mayor error.

En regresion lineal simple se puede dibujar una linea porque solo hay una variable de entrada.

## 8. Interpretacion general del resultado

El modelo logra encontrar una relacion entre `bmi` y la progresion de la enfermedad, pero el resultado no es perfecto. El valor de `R2 = 0.2334` indica que una sola variable no alcanza para explicar completamente el comportamiento de la variable objetivo.

Esto es esperable, porque la progresion de una enfermedad suele depender de muchos factores y no solamente del indice de masa corporal.

La conclusion principal es que la regresion lineal simple sirve como primera aproximacion y como modelo facil de interpretar, pero tiene una limitacion importante: usa una sola variable.

## 9. Relacion con programacion tradicional

En terminos de programacion:

- `X` funciona como input del modelo.
- `y` funciona como output esperado.
- `fit()` entrena o configura el modelo usando ejemplos.
- `predict()` recibe nuevos inputs y devuelve predicciones.
- `model` es un objeto que aprende parametros internos a partir de los datos.

La diferencia con una funcion tradicional es que no escribimos manualmente la formula completa. El modelo aprende los valores de la recta a partir del entrenamiento.

## 10. Relacion con econometria o estadistica

En estadistica, este modelo se puede interpretar como una relacion entre una variable independiente y una variable dependiente.

- Variable dependiente: `disease_progression`.
- Variable independiente: `bmi`.
- Coeficiente: indica cuanto cambia la prediccion de `y` cuando cambia `bmi`.
- Intercepto: valor base estimado cuando la variable de entrada esta en cero.
- `R2`: indica que porcentaje de la variacion de `y` es explicado por el modelo.

Como hay una sola variable independiente, el modelo es simple y facil de explicar, pero tambien limitado.

## 11. Preguntas tipicas de parcial

Que diferencia hay entre regresion y clasificacion?

Regresion predice valores numericos continuos. Clasificacion predice categorias o clases.

Que es regresion lineal simple?

Es un modelo que predice una variable numerica usando una sola variable de entrada.

Que representa `X_train`?

Representa los datos de entrada usados para entrenar el modelo.

Que representa `y_test`?

Representa los valores reales usados para evaluar las predicciones del modelo.

Que hace `fit()`?

Entrena el modelo con los datos de entrenamiento.

Que hace `predict()`?

Genera predicciones para nuevos datos de entrada.

Que significa `R2`?

Indica que tan bien el modelo explica la variacion de la variable objetivo.

## 12. Errores comunes o confusiones

- Creer que una sola variable siempre alcanza para predecir bien.
- Pensar que un `R2` bajo significa que el codigo esta mal; puede significar que faltan variables importantes.
- Evaluar el modelo con los mismos datos usados para entrenar.
- Confundir regresion con clasificacion.
- Interpretar el grafico como si todos los puntos debieran caer exactamente sobre la linea.

## 13. Cosas que aprendi

Aprendi que la regresion lineal simple permite predecir un valor numerico usando una sola variable. En este caso, el modelo usa `bmi` para intentar predecir la progresion de la enfermedad. Tambien aprendi que el modelo puede funcionar, pero si el problema depende de muchas variables, una sola columna no siempre alcanza para lograr una buena prediccion.
