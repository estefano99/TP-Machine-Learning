# Informe - Naive Bayes

## 1. Nombre del algoritmo

Naive Bayes Gaussiano, implementado con `GaussianNB`.

## 2. Objetivo del algoritmo

Naive Bayes es un algoritmo supervisado de clasificacion basado en el teorema de Bayes. Calcula la probabilidad de cada clase a partir de las variables observadas y elige la clase mas probable.

En este proyecto clasifica vinos en tres clases. Se usa la variante gaussiana porque las variables de Wine son mediciones numericas continuas.

## 3. Dataset utilizado

Dataset: `Wine`.

Fuente: `sklearn.datasets.load_wine`.

- Filas: 178.
- Variables independientes `X`: 13.
- Variable objetivo `y`: `target`.
- Columnas contando el objetivo: 14.
- Clases: `class_0`, `class_1` y `class_2`.

Las variables describen propiedades quimicas como alcohol, magnesio, flavonoides, intensidad de color y prolina. Se eligio porque permite aplicar clasificacion multiclase sobre datos numericos incluidos en `scikit-learn`.

## 4. Archivos modificados o creados

`data/loaders.py`

- Agrega `load_wine_classification()`.
- Devuelve las variables, el objetivo, los nombres de clases y datos descriptivos.

`models/classification.py`

- Agrega `run_naive_bayes_classification()`.
- Entrena `GaussianNB` y coordina metricas, matriz, grafico e informe.

`main.py`

- Conecta la opcion 10 con Naive Bayes.

Las utilidades de vista previa, metricas multiclase, heatmap e informes se reutilizan sin cambios.

## 5. Flujo de ejecucion

1. El usuario selecciona la opcion 10.
2. Se carga Wine.
3. Se separan las 13 variables `X` y la clase `y`.
4. Se realiza una division estratificada 80/20 con `random_state=42`.
5. Se crea `GaussianNB`.
6. `fit()` calcula probabilidades de clases, medias y varianzas.
7. `predict()` calcula la clase mas probable para cada vino de prueba.
8. Se obtienen metricas ponderadas y matriz de confusion.
9. Se guardan el heatmap y el informe automatico.

No se aplica escalado porque GaussianNB modela por separado la distribucion de cada variable dentro de cada clase.

## 6. Metricas obtenidas

- `Accuracy`: 0.9722.
- `Precision weighted`: 0.9744.
- `Recall weighted`: 0.9722.
- `F1-score weighted`: 0.9723.

`Accuracy` indica que el 97,22 % de los vinos fue clasificado correctamente. Las otras tres metricas se calculan para cada clase y se resumen con un promedio ponderado segun la cantidad de casos.

Matriz de confusion:

```text
              Predicho
Real          Clase 0  Clase 1  Clase 2
Clase 0            12        0        0
Clase 1             1       13        0
Clase 2             0        0       10
```

El modelo acerto 35 de 36 casos. El unico error fue un vino real de `class_1` predicho como `class_0`.

## 7. Grafico generado

Archivo: `tp_machine_learning/outputs/graphs/naive_bayes_wine.png`.

El eje X contiene las clases predichas y el eje Y las clases reales. La diagonal principal muestra los aciertos y las celdas restantes representan errores.

Las clases 0 y 2 fueron identificadas sin errores. Solo existio una confusion entre las clases 1 y 0.

## 8. Interpretacion general del resultado

Gaussian Naive Bayes obtuvo un rendimiento alto aun cuando supone que las variables son independientes entre si dentro de cada clase. Ese supuesto probablemente no se cumple de manera perfecta en las mediciones quimicas, pero el modelo puede seguir funcionando bien como aproximacion.

Su principal ventaja es que es simple, rapido y funciona con pocos datos. Como limitacion, sus supuestos pueden reducir el rendimiento cuando las variables estan muy relacionadas o no se aproximan a distribuciones normales.

## 9. Relacion con programacion tradicional

- `X` contiene las propiedades quimicas del vino.
- `y` contiene la clase esperada.
- `fit()` calcula estadisticas para cada variable y clase.
- `predict()` combina probabilidades y devuelve la clase mas probable.
- `model` almacena medias, varianzas y probabilidades previas aprendidas.

## 10. Relacion con econometria o estadistica

El modelo aplica el teorema de Bayes para estimar la probabilidad de una clase dadas las mediciones observadas. La probabilidad previa representa que tan frecuente es cada clase antes de observar las variables.

GaussianNB supone que cada variable sigue una distribucion normal dentro de cada clase y que las variables son condicionalmente independientes. Se lo llama `naive` o ingenuo precisamente por este ultimo supuesto simplificador.

## 11. Preguntas tipicas de parcial

Por que se usa `GaussianNB`?

Porque las variables del dataset son numericas continuas.

Por que se llama Naive Bayes?

Porque supone ingenuamente que las variables son independientes dada la clase.

Que aprende durante `fit()`?

Las probabilidades previas, medias y varianzas de las variables para cada clase.

Necesita escalado?

No en esta implementacion, porque estima una distribucion separada para cada variable.

Puede clasificar mas de dos clases?

Si. En Wine compara las probabilidades de tres clases.

## 12. Errores comunes o confusiones

- Pensar que Naive Bayes solo se usa para textos.
- Confundir independencia total con independencia condicionada a la clase.
- Creer que `Gaussian` significa que todo el dataset debe seguir una unica distribucion normal.
- Leer el promedio ponderado como si perteneciera a una sola clase.
- Evaluar el modelo con los mismos datos usados para entrenarlo.

## 13. Cosas que aprendi

Aprendi que Naive Bayes clasifica usando probabilidades y que su variante gaussiana es apropiada para variables numericas continuas. Tambien aprendi que un supuesto simplificador no impide necesariamente obtener buenos resultados: el modelo acerto 35 de 36 vinos y solo confundio una muestra de la clase 1.
