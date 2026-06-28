# Informe - Regresion Polinomial

## 1. Nombre del algoritmo

Regresion polinomial.

## 2. Objetivo del algoritmo

La regresion polinomial es un algoritmo de Machine Learning supervisado usado para problemas de regresion. Su objetivo es predecir un valor numerico continuo, igual que la regresion lineal, pero permitiendo representar relaciones curvas entre la variable de entrada y la variable objetivo.

En este proyecto, el modelo intenta predecir la progresion de la enfermedad en pacientes del dataset `Diabetes` usando la variable `bmi`.

La diferencia con la regresion lineal simple es que antes el modelo buscaba una recta. En cambio, la regresion polinomial transforma la variable original para poder ajustar una curva. En esta implementacion se usa un polinomio de grado 2.

## 3. Dataset utilizado

Dataset: `Diabetes`.

Fuente: incluido en `scikit-learn`, mediante `sklearn.datasets.load_diabetes`.

Datos utilizados:

- Filas: 442.
- Columnas usadas en el modelo: 2 contando la variable objetivo.
- Variable independiente original `X`: `bmi`.
- Variable objetivo `y`: `disease_progression`.
- Tipo de uso: una sola variable original transformada a caracteristicas polinomiales.
- Grado del polinomio: 2.

Se eligio este dataset porque permite comparar la regresion polinomial contra la regresion lineal simple usando el mismo dataset y la misma variable de entrada.

## 4. Archivos modificados o creados

`main.py`

- Conecta la opcion `3. Regresion polinomial` con la funcion `run_polynomial_regression()`.
- Recibe la opcion ingresada por el usuario desde consola.
- No devuelve datos; coordina el menu principal.

`data/loaders.py`

- Se reutiliza la funcion `load_diabetes_simple_regression()`.
- Carga el dataset `Diabetes`.
- Selecciona `bmi` como variable independiente.
- Devuelve `x`, `y`, informacion del dataset y un `dataframe` para vista previa.

`models/regression.py`

- Se agrego la funcion `run_polynomial_regression()`.
- Carga el dataset, separa entrenamiento y prueba, transforma `bmi` con `PolynomialFeatures`, muestra una vista previa del dataset transformado, entrena `LinearRegression`, predice, calcula metricas, genera grafico y guarda informe.
- Usa `PolynomialFeatures(degree=2, include_bias=False)` para crear caracteristicas polinomiales.

`utils/dataset_preview.py`

- Se reutiliza `print_dataset_preview()`.
- Muestra informacion basica del dataset y las primeras 5 filas.
- Se agrego o reutiliza `print_transformed_dataset_preview()`.
- Muestra las primeras filas luego de aplicar la transformacion polinomial.

`utils/metrics.py`

- Se reutilizan `calculate_regression_metrics()` y `print_regression_metrics()`.
- Calculan y muestran `MAE`, `MSE`, `RMSE` y `R2`.

`utils/plots.py`

- Se agrego `save_polynomial_regression_plot()`.
- Genera un grafico con puntos reales y una curva de prediccion.

`utils/reports.py`

- Se reutiliza `save_algorithm_report()`.
- Guarda un informe `.txt` con dataset, metricas, significado de metricas, grafico e interpretacion.

## 5. Flujo de ejecucion

1. Se ejecuta `python tp_machine_learning/main.py`.
2. Se muestra el menu principal.
3. El usuario elige la opcion `3`.
4. `main.py` llama a `run_polynomial_regression()`.
5. Se carga el dataset `Diabetes` con `load_diabetes_simple_regression()`.
6. Se separan `X` e `y`.
7. `X` contiene la variable original `bmi`.
8. `y` contiene la progresion de la enfermedad.
9. Se separan datos de entrenamiento y prueba con `train_test_split`.
10. Se transforman los datos con `PolynomialFeatures` de grado 2.
11. Se muestra una vista previa del dataset transformado, donde aparecen `bmi` y `bmi^2`.
12. Se crea un modelo `LinearRegression`.
13. Se entrena el modelo con `fit(x_train_poly, y_train)`.
14. Se generan predicciones con `predict(x_test_poly)`.
15. Se calculan las metricas de regresion.
16. Se genera el grafico `regresion_polinomial_diabetes.png`.
17. Se guarda el informe `regresion_polinomial_diabetes.txt`.

Vista previa conceptual de la transformacion:

```text
bmi        bmi^2
0.061696   0.003806
-0.051474  0.002650
0.044451   0.001976
```

La columna `bmi^2` no existe originalmente en el dataset. Es creada por `PolynomialFeatures` para que el modelo pueda aprender una curva.

## 6. Metricas obtenidas

Metricas reales de la ultima ejecucion:

- `MAE`: 52.3839.
- `MSE`: 4085.0255.
- `RMSE`: 63.9142.
- `R2`: 0.2290.

`MAE` mide el error absoluto promedio. Cuanto mas bajo, mejor. En este caso, el modelo se equivoca en promedio aproximadamente 52.38 unidades de progresion de enfermedad.

`MSE` mide el error cuadratico medio. Cuanto mas bajo, mejor. Penaliza mas los errores grandes porque eleva las diferencias al cuadrado.

`RMSE` es la raiz cuadrada del `MSE`. Cuanto mas bajo, mejor. En este caso, el error promedio en una escala similar a la variable objetivo es aproximadamente 63.91.

`R2` mide que proporcion de la variacion de los datos logra explicar el modelo. Cuanto mas cerca de 1, mejor. En este caso, `0.2290` indica que el modelo explica una parte limitada de los datos.

Comparacion con modelos anteriores:

- Regresion lineal simple: `R2 = 0.2334`.
- Regresion polinomial: `R2 = 0.2290`.
- Regresion lineal multiple: `R2 = 0.4526`.

La regresion polinomial de grado 2 no mejoro a la regresion lineal simple en esta ejecucion. Esto muestra que agregar complejidad al modelo no siempre mejora el resultado.

## 7. Grafico generado

Archivo generado:

```text
tp_machine_learning/outputs/graphs/regresion_polinomial_diabetes.png
```

El grafico muestra:

- Eje X: valor de `bmi`.
- Eje Y: progresion de la enfermedad.
- Puntos azules: valores reales del conjunto de prueba.
- Curva roja: prediccion del modelo polinomial.

Si los puntos estan cerca de la curva roja, significa que el modelo predijo bien esos casos. Si estan lejos, significa que hubo mayor error.

A diferencia de la regresion lineal simple, aca la linea roja puede curvarse porque el modelo usa una transformacion polinomial de la variable `bmi`.

## 8. Interpretacion general del resultado

El modelo polinomial intento capturar una relacion curva entre `bmi` y la progresion de la enfermedad. Sin embargo, con grado 2 el resultado fue apenas peor que el de la regresion lineal simple.

Esto no significa que el algoritmo este mal implementado. Significa que, para este conjunto de datos y usando solo `bmi`, una curva de grado 2 no aporto una mejora real.

La regresion lineal multiple siguio siendo mejor porque usa mas variables del paciente. Esto refuerza una conclusion importante: a veces mejora mas agregar informacion relevante que hacer mas complejo el modelo con la misma variable.

## 9. Relacion con programacion tradicional

En terminos de programacion:

- `X` es el input original, en este caso `bmi`.
- `PolynomialFeatures` transforma ese input y crea nuevas columnas derivadas, como `bmi²`.
- `y` es el output esperado.
- `fit()` entrena el modelo con las variables transformadas.
- `predict()` recibe datos transformados y devuelve una prediccion numerica.
- `model` es un objeto que aprende parametros internos a partir de los datos.

Una forma simple de pensarlo es que antes el modelo recibia una sola columna, y ahora recibe columnas generadas a partir de esa misma variable para poder formar una curva.

## 10. Relacion con econometria o estadistica

En estadistica, la regresion polinomial permite modelar relaciones no lineales usando una regresion lineal sobre variables transformadas.

- Variable dependiente: `disease_progression`.
- Variable independiente original: `bmi`.
- Variable transformada: `bmi²`.
- Coeficientes: indican el peso de cada termino del polinomio.
- `R2`: indica que parte de la variacion de la variable dependiente es explicada por el modelo.

Aunque el grafico sea curvo, el modelo interno sigue usando `LinearRegression`; lo que cambia es que las variables de entrada fueron transformadas antes del entrenamiento.

## 11. Preguntas tipicas de parcial

Que es la regresion polinomial?

Es un modelo de regresion que permite ajustar curvas transformando las variables de entrada con potencias.

Que diferencia hay entre regresion lineal simple y polinomial?

La lineal simple ajusta una recta. La polinomial puede ajustar una curva.

Que hace `PolynomialFeatures`?

Crea nuevas variables a partir de potencias de la variable original.

Que significa usar grado 2?

Significa que el modelo usa la variable original y su cuadrado.

Por que se sigue usando `LinearRegression`?

Porque la no linealidad se agrega en las variables de entrada, no cambiando el modelo final.

La regresion polinomial siempre mejora a la lineal?

No. En este caso no mejoro, porque el `R2` fue levemente menor.

Que representa `x_train_poly`?

Representa los datos de entrenamiento despues de aplicar la transformacion polinomial.

## 12. Errores comunes o confusiones

- Pensar que la regresion polinomial siempre es mejor que la lineal.
- Creer que se usa un algoritmo totalmente distinto, cuando en este caso se usa `LinearRegression` con variables transformadas.
- Comparar modelos sin mirar metricas reales.
- Usar grados muy altos y generar sobreajuste.
- Olvidar transformar tambien los datos de prueba antes de predecir.

## 13. Cosas que aprendi

Aprendi que la regresion polinomial permite probar una curva en lugar de una recta. Tambien aprendi que un modelo mas complejo no siempre mejora el resultado. En este caso, usando solo `bmi`, la regresion polinomial de grado 2 tuvo un `R2` apenas menor que la regresion lineal simple, mientras que la regresion multiple mejoro mas porque uso varias variables del dataset.
