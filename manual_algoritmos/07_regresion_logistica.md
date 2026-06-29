# Informe - Regresion Logistica

## 1. Nombre del algoritmo

Regresion logistica para clasificacion binaria.

## 2. Objetivo del algoritmo

La regresion logistica es un algoritmo supervisado de clasificacion. A pesar de incluir la palabra regresion, predice la probabilidad de pertenecer a una clase y luego asigna una categoria.

En este proyecto clasifica tumores como `malignant` o `benign`. Es el primer algoritmo de clasificacion del TP; los anteriores predecian valores numericos continuos.

## 3. Dataset utilizado

Dataset: `Breast Cancer Wisconsin`.

Fuente: `sklearn.datasets.load_breast_cancer`.

- Filas: 569.
- Variables independientes `X`: 30 caracteristicas numericas.
- Variable objetivo `y`: `target`.
- Columnas contando el objetivo: 31.
- Clase 0: `malignant`.
- Clase 1: `benign`.

Se eligio porque presenta un problema binario claro, es de tamaño reducido y esta incluido en `scikit-learn`.

## 4. Archivos modificados o creados

`data/loaders.py`

- Agrega `load_breast_cancer_classification()` para cargar `X`, `y`, clases y datos descriptivos.

`models/classification.py`

- Nuevo modulo para algoritmos de clasificacion.
- Agrega `run_logistic_regression()`, que coordina todo el flujo.

`utils/metrics.py`

- Agrega funciones para calcular y mostrar `accuracy`, `precision`, `recall`, `F1-score` y matriz de confusion.

`utils/plots.py`

- Agrega `save_confusion_matrix_plot()` para guardar la matriz como heatmap.

`utils/reports.py`

- Agrega las explicaciones de las metricas de clasificacion.

`main.py`

- Conecta la opcion 7 con `run_logistic_regression()`.

## 5. Flujo de ejecucion

1. El usuario selecciona la opcion 7.
2. Se carga Breast Cancer.
3. Se separan las 30 variables `X` y la clase `y`.
4. Se crea una division estratificada 80/20 con `random_state=42`.
5. Se construye un `Pipeline` con `StandardScaler` y `LogisticRegression`.
6. `fit()` aprende el escalado y entrena el clasificador usando el conjunto de entrenamiento.
7. `predict()` clasifica los casos de prueba.
8. Se calculan metricas y matriz de confusion.
9. Se guardan el heatmap y el informe automatico.

La estratificacion conserva aproximadamente la proporcion original de tumores malignos y benignos en ambos conjuntos.

## 6. Metricas obtenidas

- `Accuracy`: 0.9825.
- `Precision`: 0.9762.
- `Recall`: 0.9762.
- `F1-score`: 0.9762.

Para `precision`, `recall` y `F1-score`, la clase positiva es `malignant` (`0`).

`Accuracy` indica que el 98,25 % de todas las predicciones fue correcto.

`Precision` indica que, entre los tumores predichos como malignos, el 97,62 % era realmente maligno.

`Recall` indica que el modelo detecto el 97,62 % de los tumores malignos reales. En este problema es especialmente importante evitar que un tumor maligno sea clasificado como benigno.

`F1-score` resume el equilibrio entre precision y recall. Cuanto mas cerca de 1, mejor.

Matriz de confusion:

```text
              Predicho
Real          Maligno  Benigno
Maligno            41        1
Benigno              1       71
```

## 7. Grafico generado

Archivo: `tp_machine_learning/outputs/graphs/regresion_logistica_breast_cancer.png`.

El eje X representa la clase predicha y el eje Y la clase real. Cada celda muestra la cantidad de casos de esa combinacion.

- 41 malignos fueron detectados correctamente.
- 71 benignos fueron detectados correctamente.
- 1 maligno fue clasificado como benigno.
- 1 benigno fue clasificado como maligno.

Los valores de la diagonal principal son los aciertos. Los valores fuera de la diagonal son errores.

## 8. Interpretacion general del resultado

El modelo obtuvo un rendimiento alto y solo cometio dos errores entre 114 casos de prueba. La deteccion de tumores malignos fue muy buena, aunque existio un falso negativo maligno.

El resultado demuestra que la regresion logistica puede funcionar bien en una clasificacion binaria con variables numericas. No debe interpretarse como una herramienta medica lista para uso real: este TP es una demostracion academica y una aplicacion clinica requeriria validaciones mucho mas rigurosas.

## 9. Relacion con programacion tradicional

- `X` representa los inputs con las mediciones del tumor.
- `y` representa el output esperado: 0 o 1.
- `Pipeline` ejecuta el escalado y el modelo en orden.
- `fit()` aprende los parametros usando ejemplos conocidos.
- `predict()` devuelve la clase estimada.
- La salida se parece a una decision booleana, aunque internamente parte de una probabilidad.

## 10. Relacion con econometria o estadistica

La variable dependiente es binaria y las 30 mediciones son variables independientes. La regresion logistica modela probabilidades mediante la funcion logistica, manteniendo el resultado entre 0 y 1.

Sus coeficientes afectan el logaritmo de las probabilidades relativas de una clase. Como las variables fueron estandarizadas, trabajan sobre escalas comparables.

## 11. Preguntas tipicas de parcial

Por que se llama regresion si clasifica?

Porque calcula una funcion numerica y una probabilidad, pero la salida final se convierte en una clase.

Por que se usa `StandardScaler`?

Para que las variables trabajen en escalas comparables y facilitar el ajuste del modelo.

Que significa una division estratificada?

Que conserva aproximadamente la proporcion de cada clase al separar entrenamiento y prueba.

Que es el recall de malignos?

La proporcion de tumores malignos reales que el modelo logro detectar.

Que muestra la matriz de confusion?

Los aciertos y errores para cada combinacion de clase real y predicha.

## 12. Errores comunes o confusiones

- Pensar que la regresion logistica predice un valor continuo como la regresion lineal.
- Asumir que la clase positiva siempre es la codificada con 1; aqui se eligio `malignant` con valor 0.
- Mirar solamente `accuracy` e ignorar los falsos negativos.
- Escalar todo el dataset antes de dividirlo y provocar fuga de informacion.
- Confundir precision con recall.

## 13. Cosas que aprendi

Aprendi que la regresion logistica se utiliza para clasificar aunque su nombre incluya regresion. Tambien aprendi que `accuracy` no cuenta toda la historia: en este problema es importante revisar el recall de los tumores malignos y leer la matriz de confusion para conocer exactamente que errores cometio el modelo.
