# Informe - SVR, Regresion con Vectores de Soporte

## 1. Nombre del algoritmo

SVR, regresion con vectores de soporte, usando kernel `rbf`.

## 2. Objetivo del algoritmo

SVR es un algoritmo supervisado de regresion. Su objetivo es predecir un valor numerico continuo buscando una funcion que mantenga la mayor cantidad posible de errores dentro de un margen tolerable.

En este proyecto intenta predecir el valor medio de las viviendas de California. A diferencia de las regresiones lineales anteriores, el kernel `rbf` permite representar relaciones no lineales entre las variables y el valor objetivo.

## 3. Dataset utilizado

Dataset: `California Housing`.

Fuente: incluido en `scikit-learn`, mediante `sklearn.datasets.fetch_california_housing`.

- Dataset completo: 20.640 registros.
- Muestra utilizada: 5.000 registros.
- Variables independientes `X`: 8.
- Variable objetivo `y`: `MedHouseVal`.
- Columnas de la muestra contando el objetivo: 9.
- Semilla del muestreo: `random_state=42`.

Las variables de entrada son `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude` y `Longitude`.

Se usa una muestra fija de 5.000 registros para reducir el tiempo de entrenamiento y mantener reproducibilidad.

## 4. Archivos modificados o creados

`main.py`

- Conecta la opcion 4 con `run_svr_regression()`.

`data/loaders.py`

- Agrega `load_california_housing_sample()`.
- Carga California Housing y obtiene una muestra reproducible.
- Devuelve `X`, `y`, el DataFrame y la informacion necesaria para mostrarlo.

`models/regression.py`

- Agrega `run_svr_regression()`.
- Divide los datos, crea el `Pipeline`, entrena, predice y coordina metricas, grafico e informe.

`utils/dataset_preview.py`

- Muestra la informacion de la muestra y sus primeras cinco filas.

`utils/metrics.py`, `utils/plots.py` y `utils/reports.py`

- Reutilizan las funciones comunes de metricas, grafico e informe automatico.

## 5. Flujo de ejecucion

1. El usuario selecciona la opcion 4 del menu.
2. Se carga California Housing.
3. Se seleccionan 5.000 filas con `random_state=42`.
4. Se separan `X` e `y`.
5. Se divide la muestra en 80 % para entrenamiento y 20 % para prueba.
6. Se crea un `Pipeline` con `StandardScaler` y `SVR(kernel="rbf")`.
7. `fit()` aprende el escalado y entrena el modelo usando solamente el conjunto de entrenamiento.
8. `predict()` aplica el mismo escalado a `X_test` y genera las predicciones.
9. Se calculan las metricas.
10. Se generan el grafico y el informe automatico.

Pipeline utilizado:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVR(kernel="rbf")),
])
```

## 6. Metricas obtenidas

Resultados de la ultima ejecucion:

- `MAE`: 0.4074.
- `MSE`: 0.3620.
- `RMSE`: 0.6017.
- `R2`: 0.7320.

`MAE` indica que el error absoluto promedio es aproximadamente 0.41 unidades de `MedHouseVal`. Cuanto mas bajo, mejor.

`MSE` penaliza especialmente los errores grandes. Cuanto mas bajo, mejor.

`RMSE` expresa el error en una escala similar a la variable objetivo. El valor obtenido es aproximadamente 0.60. Cuanto mas bajo, mejor.

`R2` indica que el modelo explica aproximadamente el 73,20 % de la variacion observada en el conjunto de prueba. Cuanto mas cerca de 1, mejor.

Estas metricas no deben compararse directamente con las de Diabetes porque la variable objetivo y su escala son diferentes.

## 7. Grafico generado

Archivo: `tp_machine_learning/outputs/graphs/svr_california_housing.png`.

- Eje X: valores reales de `MedHouseVal`.
- Eje Y: valores predichos por SVR.
- Cada punto representa una vivienda o zona del conjunto de prueba.
- La linea roja representa una prediccion perfecta.

Los puntos cercanos a la linea roja representan predicciones mas precisas. Los puntos alejados muestran errores mayores.

## 8. Interpretacion general del resultado

El `R2` de 0.7320 muestra que SVR logro capturar una parte importante de la variacion del valor de las viviendas. El kernel `rbf` resulta adecuado porque las relaciones entre ubicacion, ingresos, antiguedad y valor de vivienda no tienen por que ser lineales.

El resultado tambien depende de los hiperparametros predeterminados de `SVR` y de la muestra seleccionada. Podria mejorarse ajustando `C`, `epsilon` y `gamma`, pero se mantienen los valores predeterminados para no sobrecomplicar esta primera implementacion.

## 9. Relacion con programacion tradicional

- `X` representa los datos de entrada.
- `y` representa el resultado esperado.
- `Pipeline` encadena pasos que deben ejecutarse siempre en el mismo orden.
- `fit()` configura el escalador y entrena el modelo.
- `predict()` transforma nuevos datos y devuelve predicciones.
- `pipeline` es un objeto que contiene tanto el preprocesamiento como el modelo.

## 10. Relacion con econometria o estadistica

`MedHouseVal` es la variable dependiente y las ocho columnas restantes son variables independientes. El modelo no genera coeficientes lineales faciles de interpretar porque utiliza un kernel no lineal.

El `R2` conserva su interpretacion general: mide que proporcion de la variabilidad de la variable dependiente explica el modelo sobre datos que no uso para entrenar.

## 11. Preguntas tipicas de parcial

Que significa SVR?

Support Vector Regression o regresion con vectores de soporte.

Por que se escalan las variables?

Porque SVR depende de distancias y variables con escalas grandes podrian dominar a las demas.

Que hace el kernel `rbf`?

Permite modelar relaciones no lineales entre las entradas y el objetivo.

Por que se usa un `Pipeline`?

Para asegurar que el escalado y el modelo se ejecuten juntos y evitar aplicar transformaciones inconsistentes.

Por que se usan solo 5.000 registros?

Para reducir el tiempo de entrenamiento de SVR manteniendo una muestra suficiente y reproducible.

Que garantiza `random_state=42`?

Que se seleccionen las mismas filas y se obtenga la misma division cada vez que se ejecute el programa.

## 12. Errores comunes o confusiones

- Escalar todo el dataset antes de dividirlo, causando fuga de informacion.
- Escalar `X_train` y olvidar aplicar el mismo escalador a `X_test`.
- Comparar directamente errores de datasets cuyas variables objetivo tienen escalas distintas.
- Pensar que un kernel no lineal produce coeficientes tan faciles de interpretar como una regresion lineal.
- Usar todos los registros sin considerar el costo de entrenamiento de SVR.

## 13. Cosas que aprendi

Aprendi que SVR puede representar relaciones no lineales mediante un kernel. Tambien aprendi que el escalado es importante para los modelos basados en distancias y que un `Pipeline` permite unir el escalador y el modelo para evitar errores. La muestra fija con `random_state=42` hace que la ejecucion sea mas rapida y reproducible.
