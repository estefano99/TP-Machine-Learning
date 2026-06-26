# Informe - Regresion Lineal Multiple

## 1. Nombre del algoritmo

Regresion lineal multiple.

## 2. Objetivo del algoritmo

La regresion lineal multiple es un algoritmo de Machine Learning supervisado usado para problemas de regresion. Su objetivo es predecir un valor numerico continuo usando varias variables de entrada al mismo tiempo.

En este proyecto, el modelo intenta predecir la progresion de la enfermedad en pacientes del dataset `Diabetes` usando todas las variables disponibles.

La diferencia principal con la regresion lineal simple es que el modelo anterior usaba solo `bmi`, mientras que este modelo usa varias columnas: `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5` y `s6`.

## 3. Dataset utilizado

Dataset: `Diabetes`.

Fuente: incluido en `scikit-learn`, mediante `sklearn.datasets.load_diabetes`.

Datos utilizados:

- Filas: 442.
- Columnas usadas: 11 contando la variable objetivo.
- Variables independientes `X`: `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`.
- Variable objetivo `y`: `disease_progression`.
- Tipo de uso: multiples variables de entrada.

Se eligio este dataset porque permite comparar la regresion lineal multiple contra la regresion lineal simple usando el mismo problema, pero incorporando mas informacion.

## 4. Archivos modificados o creados

`main.py`

- Conecta la opcion `2. Regresion lineal multiple` con la funcion `run_multiple_linear_regression()`.
- Recibe la opcion ingresada por el usuario.
- No devuelve datos; coordina el menu principal.

`data/loaders.py`

- Contiene la funcion `load_diabetes_multiple_regression()`.
- Carga el dataset `Diabetes`.
- Usa todas las columnas disponibles como variables independientes.
- Devuelve un diccionario con `x`, `y`, nombre del dataset, variable objetivo, variables usadas, cantidad de filas, cantidad de columnas y un `dataframe` para vista previa.

`models/regression.py`

- Contiene la funcion `run_multiple_linear_regression()`.
- Ejecuta el flujo completo del algoritmo.
- Carga datos, separa entrenamiento y prueba, entrena el modelo, predice, calcula metricas, genera grafico y guarda informe.

`utils/dataset_preview.py`

- Contiene `print_dataset_preview()`.
- Muestra informacion basica del dataset y las primeras 5 filas.

`utils/metrics.py`

- Contiene `calculate_regression_metrics()` y `print_regression_metrics()`.
- Calcula y muestra `MAE`, `MSE`, `RMSE` y `R2`.

`utils/plots.py`

- Contiene `save_real_vs_predicted_plot()`.
- Genera un grafico de valores reales contra valores predichos.

`utils/reports.py`

- Contiene `save_algorithm_report()`.
- Guarda un informe `.txt` con dataset, metricas, significado de metricas, grafico e interpretacion.

## 5. Flujo de ejecucion

1. Se ejecuta `python tp_machine_learning/main.py`.
2. Se muestra el menu principal.
3. El usuario elige la opcion `2`.
4. `main.py` llama a `run_multiple_linear_regression()`.
5. Se carga el dataset `Diabetes` con `load_diabetes_multiple_regression()`.
6. Se separan `X` e `y`.
7. `X` contiene todas las variables independientes del dataset.
8. `y` contiene la progresion de la enfermedad.
9. Se separan datos de entrenamiento y prueba con `train_test_split`.
10. Se crea un modelo `LinearRegression`.
11. Se entrena el modelo con `fit(x_train, y_train)`.
12. Se generan predicciones con `predict(x_test)`.
13. Se calculan las metricas de regresion.
14. Se genera el grafico `regresion_lineal_multiple_diabetes.png`.
15. Se guarda el informe `regresion_lineal_multiple_diabetes.txt`.

## 6. Metricas obtenidas

Metricas reales de la ultima ejecucion:

- `MAE`: 42.7941.
- `MSE`: 2900.1936.
- `RMSE`: 53.8534.
- `R2`: 0.4526.

`MAE` mide el error absoluto promedio. Cuanto mas bajo, mejor. En este caso, el modelo se equivoca en promedio aproximadamente 42.79 unidades de progresion de enfermedad.

`MSE` mide el error cuadratico medio. Cuanto mas bajo, mejor. Penaliza mas los errores grandes porque eleva las diferencias al cuadrado.

`RMSE` es la raiz cuadrada del `MSE`. Cuanto mas bajo, mejor. En este caso, el error promedio en una escala parecida a la variable objetivo es aproximadamente 53.85.

`R2` mide que proporcion de la variacion de los datos logra explicar el modelo. Cuanto mas cerca de 1, mejor. En este caso, `0.4526` indica que el modelo explica mejor los datos que la regresion lineal simple.

## 7. Grafico generado

Archivo generado:

```text
tp_machine_learning/outputs/graphs/regresion_lineal_multiple_diabetes.png
```

El grafico muestra valores reales vs valores predichos:

- Eje X: valor real de la progresion de la enfermedad.
- Eje Y: valor predicho por el modelo.
- Cada punto: un caso del conjunto de prueba.
- Linea roja: prediccion perfecta.

Si un punto esta cerca de la linea roja, significa que el valor predicho fue parecido al valor real. Si esta lejos, significa que el modelo cometio un error mayor.

En regresion lineal multiple no se grafica una recta tradicional contra una sola variable porque el modelo usa varias variables al mismo tiempo. Por eso se usa el grafico de reales vs predichos.

## 8. Interpretacion general del resultado

La regresion lineal multiple mejoro respecto a la regresion lineal simple.

Comparacion:

- Regresion lineal simple: `R2 = 0.2334`.
- Regresion lineal multiple: `R2 = 0.4526`.

Esto indica que agregar mas variables le dio mas informacion al modelo y mejoro su capacidad para explicar la progresion de la enfermedad.

Sin embargo, el modelo todavia no es perfecto. Un `R2` de `0.4526` muestra una mejora clara, pero tambien indica que todavia hay parte de la variacion que el modelo no explica. Puede deberse a que la relacion entre las variables no sea totalmente lineal o a que falten variables importantes.

La conclusion principal es que usar multiples variables puede mejorar el rendimiento, pero no garantiza una prediccion perfecta.

## 9. Relacion con programacion tradicional

En terminos de programacion:

- `X` funciona como input del modelo, pero ahora contiene varias columnas.
- `y` funciona como output esperado.
- `fit()` entrena el modelo y ajusta sus parametros internos.
- `predict()` recibe nuevos datos y devuelve una prediccion numerica.
- `model` es un objeto que aprende una formula a partir de los datos.

La diferencia con una funcion tradicional es que no programamos manualmente todos los coeficientes. El algoritmo los aprende durante el entrenamiento.

## 10. Relacion con econometria o estadistica

En estadistica, la regresion lineal multiple se interpreta como una relacion entre una variable dependiente y varias variables independientes.

- Variable dependiente: `disease_progression`.
- Variables independientes: `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`.
- Coeficientes: indican el peso o influencia estimada de cada variable en la prediccion.
- Intercepto: valor base estimado por el modelo.
- `R2`: indica que proporcion de la variacion de la variable dependiente es explicada por el conjunto de variables independientes.

Agregar mas variables puede mejorar el modelo porque le da mas informacion, pero tambien hay que tener cuidado: agregar variables irrelevantes no siempre mejora realmente la calidad del modelo.

## 11. Preguntas tipicas de parcial

Que diferencia hay entre regresion lineal simple y multiple?

La simple usa una sola variable independiente. La multiple usa varias variables independientes.

Que representa `X_train`?

Representa las variables de entrada usadas para entrenar el modelo.

Que representa `y_test`?

Representa los valores reales usados para evaluar las predicciones.

Que hace `fit()`?

Entrena el modelo ajustando sus parametros internos.

Que hace `predict()`?

Genera predicciones a partir de nuevos datos de entrada.

Que significa `R2`?

Indica que tan bien el modelo explica la variacion de la variable objetivo.

Por que el grafico de regresion multiple no muestra una recta tradicional?

Porque el modelo usa muchas variables de entrada y no se puede representar facilmente en un plano de dos dimensiones como una recta simple.

Por que mejoro el resultado respecto a la regresion simple?

Porque el modelo recibio mas informacion al usar todas las variables del dataset.

## 12. Errores comunes o confusiones

- Creer que la linea roja del grafico es una recta de regresion tradicional; en este caso representa prediccion perfecta.
- Pensar que hay puntos reales y puntos predichos separados; cada punto contiene ambos valores para un mismo caso.
- Creer que agregar mas variables siempre mejora el modelo.
- Evaluar el modelo con los mismos datos con los que se entreno.
- Confundir regresion multiple con clasificacion multiclase.

## 13. Cosas que aprendi

Aprendi que la regresion lineal multiple permite usar varias columnas como entrada para predecir un valor numerico. Tambien aprendi que, cuando hay muchas variables, no se puede mostrar el modelo como una recta simple. Por eso usamos un grafico de valores reales vs predichos, donde cada punto representa un caso y la linea roja muestra como seria una prediccion perfecta.
