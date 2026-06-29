# Informe - SVM para Clasificacion

## 1. Nombre del algoritmo

Maquina de vectores de soporte para clasificacion, `Support Vector Machine` o SVM.

## 2. Objetivo del algoritmo

SVM es un algoritmo supervisado que busca una frontera capaz de separar clases dejando el mayor margen posible entre ellas. Con un kernel puede construir fronteras no lineales.

En este proyecto clasifica tumores como `malignant` o `benign`. Se compara directamente con regresion logistica porque ambos utilizan el mismo dataset y la misma division de datos.

## 3. Dataset utilizado

Dataset: `Breast Cancer Wisconsin`.

Fuente: `sklearn.datasets.load_breast_cancer`.

- Filas: 569.
- Variables independientes `X`: 30.
- Variable objetivo `y`: `target`.
- Clase 0: `malignant`.
- Clase 1: `benign`.

Sus variables numericas son adecuadas para SVM. Reutilizar este dataset permite comparar el modelo con regresion logistica en igualdad de condiciones.

## 4. Archivos modificados o creados

`models/classification.py`

- Agrega `run_svm_classification()`.
- Construye un Pipeline con escalado y `SVC`.
- Coordina entrenamiento, metricas, matriz, grafico e informe.

`main.py`

- Conecta la opcion 9 con SVM.

`data/loaders.py`

- Reutiliza `load_breast_cancer_classification()`.

Las utilidades de vista previa, metricas, graficos e informes se reutilizan sin cambios exclusivos para SVM.

## 5. Flujo de ejecucion

1. El usuario selecciona la opcion 9.
2. Se carga Breast Cancer.
3. Se separan `X` e `y`.
4. Se realiza una division estratificada 80/20 con `random_state=42`.
5. Se crea un `Pipeline` con `StandardScaler` y `SVC(kernel="rbf")`.
6. `fit()` escala el entrenamiento y aprende la frontera de separacion.
7. `predict()` clasifica los casos de prueba.
8. Se calculan metricas y matriz de confusion.
9. Se guardan el heatmap y el informe automatico.

## 6. Metricas obtenidas

- `Accuracy`: 0.9825.
- `Precision`: 0.9762.
- `Recall`: 0.9762.
- `F1-score`: 0.9762.

La clase positiva usada para `precision`, `recall` y `F1-score` es `malignant` (`0`).

`Accuracy` indica que el 98,25 % de todos los casos fue clasificado correctamente. `Precision` indica que el 97,62 % de los casos predichos como malignos era realmente maligno. `Recall` indica que se detecto el 97,62 % de los malignos reales. `F1-score` resume el equilibrio entre precision y recall.

Matriz de confusion:

```text
              Predicho
Real          Maligno  Benigno
Maligno            41        1
Benigno              1       71
```

SVM y regresion logistica obtuvieron las mismas metricas y matriz sobre esta division. Esto significa que produjeron las mismas clases para los casos de prueba, no que sus mecanismos internos sean iguales.

## 7. Grafico generado

Archivo: `tp_machine_learning/outputs/graphs/svm_clasificacion_breast_cancer.png`.

El eje X representa la clase predicha y el eje Y la clase real. La diagonal contiene 41 malignos y 71 benignos correctamente identificados. Fuera de la diagonal aparece un error de cada tipo.

## 8. Interpretacion general del resultado

SVM obtuvo un rendimiento alto: detecto 41 de los 42 tumores malignos del conjunto de prueba y cometio solo dos errores totales.

Aunque el kernel RBF permite fronteras no lineales, en esta division no mejoro las predicciones finales de regresion logistica. Esto muestra que un modelo mas flexible no siempre produce mejores resultados observables. La comparacion rigurosa requeriria validacion cruzada y ajuste de hiperparametros.

Este ejercicio es academico y no constituye una herramienta de diagnostico medico.

## 9. Relacion con programacion tradicional

- `X` contiene las mediciones de entrada.
- `y` contiene la clase esperada.
- `Pipeline` encadena escalado y clasificacion.
- `fit()` aprende la frontera entre clases.
- `predict()` indica de que lado de esa frontera queda un nuevo caso.
- Los vectores de soporte son ejemplos importantes para definir la frontera.

## 10. Relacion con econometria o estadistica

La variable dependiente es binaria y las 30 mediciones son variables independientes. A diferencia de regresion logistica, SVM no se centra en modelar directamente una probabilidad; busca una frontera con margen amplio.

El kernel RBF transforma implicitamente la representacion de los datos para permitir separaciones no lineales. El escalado es importante porque el modelo depende de distancias.

## 11. Preguntas tipicas de parcial

Que significa SVM?

Support Vector Machine o maquina de vectores de soporte.

Que busca maximizar SVM?

El margen entre la frontera de decision y los casos mas cercanos de cada clase.

Que son los vectores de soporte?

Los ejemplos cercanos a la frontera que influyen especialmente en su posicion.

Que hace el kernel RBF?

Permite construir fronteras no lineales mediante similitudes basadas en distancia.

Por que se usa `StandardScaler`?

Porque las diferencias de escala afectarian las distancias y el kernel.

## 12. Errores comunes o confusiones

- Confundir SVM de clasificacion con SVR de regresion.
- Olvidar escalar variables medidas en unidades diferentes.
- Pensar que metricas iguales significan que dos algoritmos son identicos.
- Considerar benigno como clase positiva sin revisar la codificacion del dataset.
- Evaluar un modelo medico solo mediante `accuracy`.

## 13. Cosas que aprendi

Aprendi que SVM intenta separar las clases con el mayor margen posible y que el kernel RBF permite fronteras no lineales. Tambien aprendi que dos modelos diferentes pueden obtener exactamente las mismas predicciones en un conjunto de prueba. En este caso SVM y regresion logistica tuvieron el mismo resultado, aunque funcionan de manera distinta.
