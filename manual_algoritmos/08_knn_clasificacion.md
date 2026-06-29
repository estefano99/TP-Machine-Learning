# Informe - K Vecinos Mas Cercanos (KNN)

## 1. Nombre del algoritmo

K vecinos mas cercanos, `K-Nearest Neighbors` o KNN.

## 2. Objetivo del algoritmo

KNN es un algoritmo supervisado que clasifica un caso observando las clases de sus vecinos mas cercanos. La clase que obtiene mas votos se asigna al nuevo caso.

En este proyecto clasifica flores Iris como `setosa`, `versicolor` o `virginica`. A diferencia de la regresion logistica anterior, este es un problema multiclase con tres resultados posibles.

## 3. Dataset utilizado

Dataset: `Iris`.

Fuente: `sklearn.datasets.load_iris`.

- Filas: 150.
- Variables independientes `X`: 4.
- Variable objetivo `y`: `target`.
- Columnas contando el objetivo: 5.
- Clases: `setosa`, `versicolor` y `virginica`.
- Registros por clase: 50.

Las variables miden el largo y ancho de sepalos y petalos. Se eligio Iris porque es pequeño, equilibrado y didactico para una primera clasificacion multiclase.

## 4. Archivos modificados o creados

`data/loaders.py`

- Agrega `load_iris_classification()` para cargar datos, clases e informacion descriptiva.

`models/classification.py`

- Agrega `run_knn_classification()`.
- Construye el Pipeline, entrena KNN y coordina resultados, grafico e informe.

`utils/metrics.py`

- Adapta las funciones para admitir clasificacion binaria y multiclase.
- Permite usar promedio ponderado y matrices con cualquier cantidad de clases.

`main.py`

- Conecta la opcion 8 con KNN.

Las funciones comunes de vista previa, grafico e informe son reutilizadas sin crear versiones exclusivas para Iris.

## 5. Flujo de ejecucion

1. El usuario selecciona la opcion 8.
2. Se carga Iris.
3. Se separan las cuatro variables `X` y la clase `y`.
4. Se realiza una division estratificada 80/20 con `random_state=42`.
5. Se crea un `Pipeline` con `StandardScaler` y KNN usando cinco vecinos.
6. `fit()` almacena los datos de entrenamiento escalados.
7. `predict()` busca los cinco vecinos mas cercanos de cada flor de prueba.
8. La clase se decide mediante votacion.
9. Se calculan metricas y matriz de confusion.
10. Se guardan el heatmap y el informe automatico.

## 6. Metricas obtenidas

- `Accuracy`: 0.9333.
- `Precision weighted`: 0.9444.
- `Recall weighted`: 0.9333.
- `F1-score weighted`: 0.9327.

`Accuracy` indica que se clasifico correctamente el 93,33 % de las 30 flores del conjunto de prueba.

Como existen tres clases, `precision`, `recall` y `F1-score` se calculan para cada especie y se resumen mediante un promedio ponderado. Cada clase aporta al resultado segun su cantidad de casos.

Matriz de confusion:

```text
              Predicho
Real          Setosa  Versicolor  Virginica
Setosa            10           0          0
Versicolor         0          10          0
Virginica          0           2          8
```

## 7. Grafico generado

Archivo: `tp_machine_learning/outputs/graphs/knn_iris.png`.

El eje X representa la especie predicha y el eje Y la especie real. La diagonal principal contiene los aciertos y las otras celdas muestran las confusiones.

KNN clasifico correctamente todas las flores setosa y versicolor. Dos flores virginica fueron predichas como versicolor.

## 8. Interpretacion general del resultado

El modelo obtuvo un rendimiento alto. Setosa fue facil de separar y los unicos errores aparecieron entre versicolor y virginica, especies cuyas medidas pueden ser mas parecidas.

El resultado depende de `k`, del escalado y de la division de datos. Probar muchos valores y elegir el mejor usando directamente el conjunto de prueba produciria una evaluacion poco confiable; para este TP se mantiene `k=5` como configuracion sencilla.

## 9. Relacion con programacion tradicional

- `X` representa las medidas de cada flor.
- `y` representa la especie esperada.
- `fit()` almacena y prepara los ejemplos de entrenamiento.
- `predict()` calcula distancias, selecciona vecinos y realiza una votacion.
- `k=5` significa que se consultan cinco ejemplos cercanos.

KNN aprende de una forma particular: no construye una ecuacion o un arbol, sino que conserva los datos para compararlos al predecir.

## 10. Relacion con econometria o estadistica

La especie es la variable dependiente categorica y las cuatro medidas son variables independientes. KNN es un metodo no parametrico: no supone una ecuacion lineal ni estima una cantidad fija de coeficientes.

La cercania se calcula mediante distancia. Por eso el escalado es importante: evita que una variable domine solo por expresarse con numeros mayores.

## 11. Preguntas tipicas de parcial

Que significa la K en KNN?

La cantidad de vecinos considerados para decidir la clase.

Como clasifica una nueva observacion?

Busca los vecinos mas cercanos y elige la clase con mas votos.

Por que se escalan las variables?

Porque KNN calcula distancias y todas las variables deben aportar en escalas comparables.

Que significa promedio `weighted`?

Que la metrica de cada clase se pondera segun su cantidad de casos.

Por que se usa division estratificada?

Para conservar la proporcion de las tres especies en entrenamiento y prueba.

## 12. Errores comunes o confusiones

- Creer que KNN crea una formula durante `fit()`.
- Olvidar escalar las variables antes de calcular distancias.
- Elegir `k` usando el conjunto de prueba repetidamente.
- Confundir cantidad de vecinos con cantidad de clases.
- Leer la matriz de confusion sin distinguir filas reales y columnas predichas.

## 13. Cosas que aprendi

Aprendi que KNN clasifica un caso mirando ejemplos parecidos y realizando una votacion. Tambien aprendi que el escalado es necesario cuando un algoritmo utiliza distancias y que, para resumir metricas de varias clases, se puede usar un promedio ponderado. La matriz mostro que versicolor y virginica son mas faciles de confundir que setosa.
