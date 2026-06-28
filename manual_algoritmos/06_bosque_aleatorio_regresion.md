# Informe - Bosque Aleatorio para Regresion

## 1. Nombre del algoritmo

Bosque aleatorio para regresion, tambien llamado `Random Forest Regressor`.

## 2. Objetivo del algoritmo

Es un algoritmo supervisado que predice valores numericos continuos. Entrena varios arboles de decision con diferentes muestras y subconjuntos de variables, y calcula el promedio de sus predicciones.

En este proyecto intenta predecir el valor medio de las viviendas de California. Su principal diferencia con el algoritmo anterior es que combina 100 arboles en lugar de depender de uno solo.

## 3. Dataset utilizado

Dataset: `California Housing`, incluido en `scikit-learn`.

- Dataset completo: 20.640 registros.
- Muestra utilizada: 5.000 registros con `random_state=42`.
- Variables independientes `X`: 8.
- Variable objetivo `y`: `MedHouseVal`.
- Division: 80 % entrenamiento y 20 % prueba.

Se reutiliza exactamente la misma muestra y division empleadas con SVR y el arbol individual para comparar los modelos en igualdad de condiciones.

## 4. Archivos modificados o creados

`models/regression.py`

- Agrega `run_random_forest_regression()`.
- Entrena `RandomForestRegressor` con 100 arboles.
- Coordina predicciones, metricas, grafico e informe.

`main.py`

- Conecta la opcion 6 con la nueva funcion.

`data/loaders.py`

- Reutiliza `load_california_housing_sample()` para obtener la misma muestra.

`utils/dataset_preview.py`, `utils/metrics.py`, `utils/plots.py` y `utils/reports.py`

- Reutilizan las funciones comunes ya existentes.

## 5. Flujo de ejecucion

1. El usuario selecciona la opcion 6.
2. Se carga la muestra fija de California Housing.
3. Se separan `X` e `y`.
4. Se divide la muestra en entrenamiento y prueba.
5. Se crea un bosque de 100 arboles.
6. `fit()` entrena los arboles usando variaciones de los datos y variables.
7. `predict()` promedia sus predicciones para cada caso de prueba.
8. Se calculan las metricas.
9. Se generan el grafico y el informe automatico.

Configuracion principal:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
```

No se aplica escalado porque los arboles crean cortes sobre cada variable y no calculan distancias entre variables.

## 6. Metricas obtenidas

- `MAE`: 0.3962.
- `MSE`: 0.3454.
- `RMSE`: 0.5877.
- `R2`: 0.7443.

`MAE` indica que el error absoluto promedio fue aproximadamente 0.40 unidades de `MedHouseVal`. Cuanto mas bajo, mejor.

`MSE` penaliza especialmente los errores grandes. `RMSE` expresa el error en una escala similar a la variable objetivo. Ambos deben ser lo mas bajos posible.

`R2` indica que el bosque explica aproximadamente el 74,43 % de la variacion observada en el conjunto de prueba. Cuanto mas cerca de 1, mejor.

Comparacion sobre la misma muestra:

- Arbol de decision: `R2 = 0.5984`.
- SVR: `R2 = 0.7320`.
- Bosque aleatorio: `R2 = 0.7443`.

El bosque aleatorio obtuvo el mejor resultado de los tres.

## 7. Grafico generado

Archivo: `tp_machine_learning/outputs/graphs/bosque_aleatorio_regresion_california_housing.png`.

- Eje X: valor real de `MedHouseVal`.
- Eje Y: valor predicho.
- Cada punto combina el valor real y predicho de un caso.
- La linea roja representa una prediccion perfecta.

Cuanto mas cerca queda un punto de la linea roja, menor fue el error de esa prediccion.

## 8. Interpretacion general del resultado

El bosque mejoro al arbol individual porque promedia las respuestas de muchos arboles y reduce la influencia de los errores particulares de cada uno. Tambien supero ligeramente a SVR con la configuracion utilizada.

Como limitaciones, necesita mas memoria y tiempo que un solo arbol, y sus decisiones son mas dificiles de explicar porque ya no existe un unico conjunto de reglas.

## 9. Relacion con programacion tradicional

- `X` representa los inputs.
- `y` representa el output esperado.
- Cada arbol se parece a un conjunto de condiciones `if`.
- `fit()` construye 100 arboles.
- `predict()` consulta todos los arboles y promedia sus respuestas.
- `model` encapsula el conjunto completo de arboles.

## 10. Relacion con econometria o estadistica

`MedHouseVal` es la variable dependiente y las ocho caracteristicas son las variables independientes. El modelo no produce una ecuacion lineal con coeficientes; aproxima la relacion mediante muchas particiones de los datos.

Promediar varios modelos reduce la varianza respecto de un arbol individual. El `R2` permite medir la proporcion de variabilidad explicada sobre datos de prueba.

## 11. Preguntas tipicas de parcial

Que es un bosque aleatorio?

Es un conjunto de arboles de decision cuyas predicciones se combinan.

Que significa `n_estimators=100`?

Que el bosque contiene 100 arboles.

Como obtiene la prediccion final en regresion?

Calcula el promedio de las predicciones de los arboles.

Por que puede mejorar a un solo arbol?

Porque reduce la variabilidad y la dependencia de las decisiones de un unico modelo.

Para que sirve `random_state=42`?

Para reproducir la construccion aleatoria del bosque y obtener los mismos resultados.

Que hace `n_jobs=-1`?

Permite usar todos los nucleos disponibles del procesador para entrenar.

## 12. Errores comunes o confusiones

- Pensar que es un unico arbol muy grande.
- Creer que siempre necesita escalado.
- Comparar sus metricas con modelos evaluados sobre datos diferentes.
- Pensar que aumentar indefinidamente la cantidad de arboles siempre produce grandes mejoras.
- Olvidar fijar `random_state` y obtener resultados diferentes entre ejecuciones.

## 13. Cosas que aprendi

Aprendi que Random Forest combina muchos arboles y promedia sus predicciones. Esto permite obtener un resultado mas estable y, en esta prueba, mas preciso que un solo arbol. Tambien aprendi que esta mejora tiene un costo: el modelo utiliza mas recursos y es menos sencillo de interpretar.
