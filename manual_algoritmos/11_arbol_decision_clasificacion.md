# Informe - Arbol de Decision para Clasificacion

## 1. Nombre del algoritmo

Arbol de decision para clasificacion.

## 2. Objetivo del algoritmo

Es un algoritmo supervisado que clasifica observaciones mediante reglas sucesivas. Cada nodo formula una condicion sobre una variable, cada rama representa una respuesta y cada hoja contiene una clase final.

En este proyecto clasifica flores como `setosa`, `versicolor` o `virginica`. Se compara con KNN usando exactamente el mismo dataset y la misma division.

## 3. Dataset utilizado

Dataset: `Iris`.

Fuente: `sklearn.datasets.load_iris`.

- Filas: 150.
- Variables independientes `X`: 4.
- Variable objetivo `y`: `target`.
- Columnas contando el objetivo: 5.
- Clases: `setosa`, `versicolor` y `virginica`.

Las variables representan largo y ancho de sepalos y petalos. Iris permite construir reglas sencillas y observar como un arbol separa tres clases.

## 4. Archivos modificados o creados

`models/classification.py`

- Agrega `run_decision_tree_classification()`.
- Entrena el arbol y coordina metricas, matriz, grafico e informe.

`main.py`

- Conecta la opcion 11 con el nuevo algoritmo.

`data/loaders.py`

- Reutiliza `load_iris_classification()`.

Las utilidades de vista previa, metricas multiclase, heatmap e informes se reutilizan sin cambios.

## 5. Flujo de ejecucion

1. El usuario selecciona la opcion 11.
2. Se carga Iris.
3. Se separan las cuatro variables `X` y la clase `y`.
4. Se realiza una division estratificada 80/20 con `random_state=42`.
5. Se crea `DecisionTreeClassifier(max_depth=3, random_state=42)`.
6. `fit()` busca variables y valores de corte que separen las clases.
7. `predict()` recorre las reglas para cada flor de prueba.
8. Se calculan metricas ponderadas y matriz de confusion.
9. Se guardan el heatmap y el informe automatico.

No se aplica escalado porque cambiar la escala no altera el orden de los valores ni los cortes disponibles para el arbol.

## 6. Metricas obtenidas

- `Accuracy`: 0.9667.
- `Precision weighted`: 0.9697.
- `Recall weighted`: 0.9667.
- `F1-score weighted`: 0.9666.

`Accuracy` indica que se clasifico correctamente el 96,67 % de las flores de prueba. Las otras metricas resumen el rendimiento de las tres especies mediante un promedio ponderado.

Matriz de confusion:

```text
              Predicho
Real          Setosa  Versicolor  Virginica
Setosa            10           0          0
Versicolor         0           9          1
Virginica          0           0         10
```

El modelo acerto 29 de 30 flores. Solo clasifico una `versicolor` como `virginica`.

## 7. Grafico generado

Archivo: `tp_machine_learning/outputs/graphs/arbol_decision_clasificacion_iris.png`.

El eje X representa la clase predicha y el eje Y la clase real. La diagonal contiene los aciertos y las otras celdas muestran errores.

Setosa y virginica fueron clasificadas correctamente en todos los casos de prueba. La unica confusion ocurrio entre versicolor y virginica.

## 8. Interpretacion general del resultado

El arbol obtuvo `accuracy 0.9667`, superior al `0.9333` de KNN sobre la misma division. Ademas, no necesito escalado y sus reglas pueden inspeccionarse con mayor facilidad.

La profundidad maxima de 3 mantiene el modelo compacto y reduce el riesgo de memorizar el entrenamiento. Sin embargo, el resultado pertenece a una unica division; una comparacion mas robusta requeriria validacion cruzada.

## 9. Relacion con programacion tradicional

- `X` contiene las medidas de la flor.
- `y` contiene la especie esperada.
- Cada nodo se parece a una condicion `if`.
- Cada rama se parece a una alternativa del flujo.
- `fit()` aprende automaticamente condiciones y valores de corte.
- `predict()` recorre las reglas hasta llegar a una hoja.

## 10. Relacion con econometria o estadistica

La especie es una variable dependiente categorica y las medidas son variables independientes. El arbol divide el espacio de datos buscando reducir la mezcla de clases en cada grupo.

Para elegir cortes puede utilizar medidas de impureza como Gini. Un nodo puro contiene observaciones de una sola clase; uno impuro contiene varias clases.

## 11. Preguntas tipicas de parcial

Que representa un nodo del arbol?

Una condicion sobre una variable de entrada.

Que representa una hoja?

La clase final predicha por el modelo.

Que significa `max_depth=3`?

Que puede haber como maximo tres niveles de decisiones desde la raiz.

Por que se limita la profundidad?

Para mantener reglas simples y reducir el sobreajuste.

Por que no necesita `StandardScaler`?

Porque compara valores dentro de cada variable y no calcula distancias.

## 12. Errores comunes o confusiones

- Dejar crecer el arbol sin limites y provocar sobreajuste.
- Pensar que todos los clasificadores necesitan escalado.
- Confundir una hoja con una observacion individual.
- Evaluar el modelo usando los datos de entrenamiento.
- Comparar algoritmos que usaron divisiones diferentes.

## 13. Cosas que aprendi

Aprendi que un arbol de clasificacion construye reglas parecidas a condiciones para decidir una clase. Tambien aprendi que limitar la profundidad puede mantener el modelo comprensible y que los arboles no necesitan escalado. En esta division de Iris, el arbol supero a KNN y solo cometio un error.
