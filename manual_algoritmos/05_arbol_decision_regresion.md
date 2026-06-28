# Informe - Arbol de Decision para Regresion

## 1. Nombre del algoritmo

Arbol de decision para regresion.

## 2. Objetivo del algoritmo

Es un algoritmo supervisado que predice valores numericos continuos. Divide los datos mediante preguntas sucesivas sobre sus variables y asigna una prediccion a cada grupo final, llamado hoja.

En este proyecto predice el valor medio de viviendas de California. A diferencia de SVR, no necesita escalar las variables y sus decisiones pueden representarse como reglas.

## 3. Dataset utilizado

Dataset: `California Housing`, incluido en `scikit-learn`.

- Dataset completo: 20.640 registros.
- Muestra utilizada: 5.000 registros con `random_state=42`.
- Variables independientes: 8.
- Variable objetivo: `MedHouseVal`.
- Division: 80 % entrenamiento y 20 % prueba.

Se utiliza la misma muestra que en SVR para que la comparacion entre ambos algoritmos sea justa y reproducible.

## 4. Archivos modificados o creados

`data/loaders.py`

- El cargador compartido pasa a llamarse `load_california_housing_sample()`.
- Devuelve una muestra reproducible y sus datos descriptivos.

`models/regression.py`

- Agrega `run_decision_tree_regression()`.
- Crea y entrena `DecisionTreeRegressor(max_depth=5, random_state=42)`.
- Coordina la vista previa, metricas, grafico e informe.

`main.py`

- Conecta la opcion 5 con la nueva funcion.

`utils/dataset_preview.py`, `utils/metrics.py`, `utils/plots.py` y `utils/reports.py`

- Reutilizan las funciones comunes ya existentes.

## 5. Flujo de ejecucion

1. El usuario selecciona la opcion 5.
2. Se carga la muestra de California Housing.
3. Se separan las variables `X` y el objetivo `y`.
4. Se divide la muestra en entrenamiento y prueba con `random_state=42`.
5. Se crea un arbol con profundidad maxima 5.
6. `fit()` aprende reglas usando `X_train` e `y_train`.
7. `predict()` obtiene resultados para `X_test`.
8. Se calculan las metricas.
9. Se generan el grafico y el informe automatico.

No se usa `StandardScaler`: cambiar la escala de una variable no modifica el orden de sus valores ni los cortes que puede evaluar el arbol.

## 6. Metricas obtenidas

- `MAE`: 0.5203.
- `MSE`: 0.5424.
- `RMSE`: 0.7365.
- `R2`: 0.5984.

El `MAE` indica un error absoluto promedio de aproximadamente 0.52 unidades de `MedHouseVal`. Cuanto menor, mejor.

El `MSE` penaliza con mayor fuerza los errores grandes. El `RMSE` expresa ese error en una escala similar a la variable objetivo. Ambos deben ser lo mas bajos posible.

El `R2` indica que el modelo explica aproximadamente el 59,84 % de la variacion del conjunto de prueba. Cuanto mas cercano a 1, mejor.

Comparacion sobre la misma muestra:

- SVR: `R2 = 0.7320`.
- Arbol de decision: `R2 = 0.5984`.

SVR obtuvo mayor precision en esta configuracion.

## 7. Grafico generado

Archivo: `tp_machine_learning/outputs/graphs/arbol_decision_regresion_california_housing.png`.

- Eje X: valores reales de `MedHouseVal`.
- Eje Y: valores predichos.
- Cada punto representa un caso del conjunto de prueba.
- La linea roja representa una prediccion perfecta.

Los puntos mas cercanos a la linea roja corresponden a predicciones mas precisas.

## 8. Interpretacion general del resultado

El arbol logro explicar una parte considerable de la variacion, aunque rindio menos que SVR. `max_depth=5` evita que el modelo cree demasiadas reglas y memorice el entrenamiento, pero tambien limita los detalles que puede aprender.

Esta diferencia muestra que no existe un algoritmo que siempre sea mejor. El rendimiento depende de los datos y de sus hiperparametros. El arbol conserva como ventaja que sus decisiones son mas faciles de explicar.

## 9. Relacion con programacion tradicional

- `X` es el input.
- `y` es el output esperado.
- Cada nodo se parece a una condicion `if` sobre una variable.
- `fit()` aprende automaticamente las condiciones y valores de corte.
- `predict()` recorre las reglas hasta una hoja y devuelve su valor.

## 10. Relacion con econometria o estadistica

`MedHouseVal` es la variable dependiente y las ocho caracteristicas son variables independientes. El arbol no estima una ecuacion con coeficientes como la regresion lineal; divide el espacio de datos en regiones y predice un valor para cada region.

El `R2` permite evaluar que parte de la variabilidad logra explicar sobre datos no usados durante el entrenamiento.

## 11. Preguntas tipicas de parcial

Que hace un arbol de regresion?

Divide los datos mediante reglas y predice un valor numerico en cada hoja.

Que significa `max_depth=5`?

Que desde la raiz hasta una hoja puede haber como maximo cinco niveles de decisiones.

Por que se limita la profundidad?

Para reducir el riesgo de sobreajuste.

Por que no se usa `StandardScaler`?

Porque el arbol toma decisiones comparando valores dentro de cada variable y no calcula distancias.

Que diferencia tiene con un arbol de clasificacion?

El de regresion predice numeros continuos; el de clasificacion predice clases.

## 12. Errores comunes o confusiones

- Dejar crecer el arbol sin limites y provocar sobreajuste.
- Creer que todos los algoritmos necesitan escalado.
- Evaluar el modelo con los mismos datos usados para entrenarlo.
- Confundir las hojas con clases: en regresion contienen predicciones numericas.
- Comparar modelos usando muestras o divisiones diferentes.

## 13. Cosas que aprendi

Aprendi que un arbol de regresion construye reglas parecidas a condiciones para predecir un numero. No necesita escalar las variables, pero hay que controlar su profundidad para evitar que memorice los datos. Al usar la misma muestra comprobe que SVR fue mas preciso, aunque el arbol resulta mas sencillo de interpretar.
