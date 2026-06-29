# Memoria de Desarrollo - TP Machine Learning

Este archivo funciona como guia de trabajo del proyecto. Resume que pide el TP, que algoritmos vamos a implementar, como dividimos el desarrollo por etapas y que decisiones tecnicas venimos tomando.

## Objetivo general

Desarrollar una aplicacion de consola en Python para ejecutar algoritmos de Machine Learning con datasets distintos a los usados en los ejemplos de la playlist.

Cada algoritmo debe:

- Cargar un dataset.
- Mostrar una vista previa chica del dataset.
- Separar datos de entrenamiento y prueba.
- Entrenar un modelo.
- Calcular metricas.
- Generar un grafico.
- Guardar un informe con resultados.
- Generar un informe Markdown para el manual de estudio siguiendo `ALGORITHM_REPORT_TEMPLATE.md`.

## Criterios del proyecto

- Priorizar datasets incluidos en `scikit-learn`.
- Mantener una arquitectura simple y modular.
- Evitar dependencias externas innecesarias.
- Mostrar solo las primeras 5 filas del dataset para no ensuciar la consola.
- En algoritmos con transformaciones, mostrar tambien una vista previa chica del dataset transformado.
- Usar `random_state=42` para resultados reproducibles.
- Guardar graficos en `tp_machine_learning/outputs/graphs/`.
- Guardar informes en `tp_machine_learning/outputs/reports/`.
- Guardar informes manuales de estudio en `manual_algoritmos/`.
- Crear un informe Markdown cada vez que se agregue un algoritmo nuevo.

## Estructura del proyecto

```text
tp_machine_learning/
  main.py
  requirements.txt

  data/
    loaders.py

  models/
    classification.py
    regression.py

  utils/
    dataset_preview.py
    metrics.py
    plots.py
    reports.py

  outputs/
    graphs/
    reports/
```

## Etapas de desarrollo

### Etapa 1: Menu minimo funcionando

Estado: completada.

Incluye:

- Menu de consola.
- Opcion `0. Salir`.
- Opciones para todos los algoritmos.
- Validacion de opciones invalidas.
- Mensaje `Funcion en desarrollo` para opciones no implementadas.

### Etapa 2: Primer algoritmo completo

Estado: completada.

Algoritmo implementado:

- Regresion lineal simple.

Dataset:

- `Diabetes`, usando solo la variable `bmi`.

Incluye:

- Carga de datos.
- Vista previa del dataset.
- Entrenamiento con `LinearRegression`.
- Metricas de regresion.
- Grafico de puntos reales y linea de prediccion.
- Informe `.txt`.

### Etapa 3: Reutilizacion de funciones comunes

Estado: completada.

Ya existen funciones comunes para:

- Vista previa de datasets: `utils/dataset_preview.py`.
- Metricas de regresion: `utils/metrics.py`.
- Graficos: `utils/plots.py`.
- Informes: `utils/reports.py`.
- Metricas de clasificacion: `utils/metrics.py`.
- Matriz de confusion: `utils/plots.py`.
- Algoritmos de clasificacion: `models/classification.py`.

### Etapa 4: Agregar el resto de algoritmos

Estado: en progreso.

Algoritmos de regresion:

- [x] Regresion lineal simple.
- [x] Regresion lineal multiple.
- [x] Regresion polinomial.
- [x] SVR, regresion con vectores de soporte.
- [x] Arbol de decision para regresion.
- [x] Bosque aleatorio para regresion.

Algoritmos de clasificacion:

- [x] Regresion logistica.
- [x] K vecinos mas cercanos, KNN.
- [x] SVM para clasificacion.
- [x] Naive Bayes.
- [x] Arbol de decision para clasificacion.
- [ ] Bosque aleatorio para clasificacion.

### Etapa 5: Graficos e informes

Estado: iniciada.

Ya implementado:

- Grafico para regresion lineal simple.
- Grafico de valores reales vs valores predichos para regresion multiple.
- Grafico de curva para regresion polinomial.
- Grafico de valores reales vs valores predichos para SVR.
- Grafico de valores reales vs valores predichos para arbol de decision.
- Grafico de valores reales vs valores predichos para bosque aleatorio.
- Heatmap de matriz de confusion para regresion logistica.
- Heatmap de matriz de confusion multiclase para KNN.
- Heatmap de matriz de confusion para SVM.
- Heatmap de matriz de confusion multiclase para Naive Bayes.
- Heatmap de matriz de confusion multiclase para arbol de decision.
- Informes con:
  - Algoritmo.
  - Dataset.
  - Motivo de eleccion del dataset.
  - Datos utilizados.
  - Metricas.
  - Significado de cada metrica.
  - Ruta del grafico generado.
  - Interpretacion breve.
- Plantilla de informes manuales: `ALGORITHM_REPORT_TEMPLATE.md`.

Informes manuales generados:

- `manual_algoritmos/01_regresion_lineal_simple.md`.
- `manual_algoritmos/02_regresion_lineal_multiple.md`.
- `manual_algoritmos/03_regresion_polinomial.md`.
- `manual_algoritmos/04_svr_regresion.md`.
- `manual_algoritmos/05_arbol_decision_regresion.md`.
- `manual_algoritmos/06_bosque_aleatorio_regresion.md`.
- `manual_algoritmos/07_regresion_logistica.md`.
- `manual_algoritmos/08_knn_clasificacion.md`.
- `manual_algoritmos/09_svm_clasificacion.md`.
- `manual_algoritmos/10_naive_bayes.md`.
- `manual_algoritmos/11_arbol_decision_clasificacion.md`.

Pendiente:

- Interpretaciones especificas para cada algoritmo.

## Algoritmos y datasets definidos

### Regresion lineal simple

Estado: implementado.

Dataset:

- `Diabetes`.

Variables:

- Entrada: `bmi`.
- Objetivo: `disease_progression`.

Motivo:

- Permite mostrar claramente una regresion con una sola variable.

### Regresion lineal multiple

Estado: implementado.

Dataset:

- `Diabetes`.

Variables:

- Entrada: todas las variables del dataset.
- Objetivo: `disease_progression`.

Motivo:

- Permite comparar contra la regresion lineal simple usando el mismo problema, pero con mas informacion.

### Regresion polinomial

Estado: implementado.

Dataset:

- `Diabetes`, usando `bmi`.

Motivo:

- Permite comparar una curva polinomial contra la regresion lineal simple usando el mismo dataset.

### SVR

Estado: implementado.

Dataset:

- `California Housing`, usando una muestra fija de 5.000 registros.

Motivo:

- Contiene multiples variables numericas relacionadas con el valor de las viviendas.
- Permite aplicar SVR a un problema de regresion con relaciones no lineales.

Decisiones tecnicas:

- La muestra se obtiene con `random_state=42` para mantener reproducibilidad.
- Se usa `Pipeline` con `StandardScaler` y `SVR(kernel="rbf")`.
- El escalado queda encapsulado junto al modelo y se aplica correctamente al entrenar y predecir.

### Arbol de decision para regresion

Estado: implementado.

Dataset:

- `California Housing`, usando la misma muestra fija de 5.000 registros que SVR.

Motivo:

- Permite comparar el arbol y SVR sobre exactamente los mismos datos.
- Sus variables numericas permiten crear reglas de decision para predecir valores continuos.

Decisiones tecnicas:

- Se usa `DecisionTreeRegressor(max_depth=5, random_state=42)`.
- La profundidad se limita para reducir el sobreajuste.
- No se aplica escalado porque el arbol evalua cortes dentro de cada variable.

### Bosque aleatorio para regresion

Estado: implementado.

Dataset:

- `California Housing`, usando la misma muestra fija de 5.000 registros.

Motivo:

- Permite comparar directamente un bosque de 100 arboles contra el arbol individual y SVR.

Decisiones tecnicas:

- Se usa `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`.
- No se aplica escalado porque los arboles no dependen de distancias entre variables.
- `n_jobs=-1` permite utilizar los nucleos disponibles para entrenar los arboles.

### Regresion logistica

Estado: implementado.

Dataset:

- `Breast Cancer`.

Motivo:

- Es un problema de clasificacion binaria claro y viene incluido en `scikit-learn`.

Decisiones tecnicas:

- Se usa division estratificada con `random_state=42`.
- Se usa un `Pipeline` con `StandardScaler` y `LogisticRegression`.
- Se considera `malignant` (`0`) como clase positiva para `precision`, `recall` y `F1-score`.
- Se genera una matriz de confusion con las clases `malignant` y `benign`.

### KNN

Estado: implementado.

Dataset:

- `Iris`.

Motivo:

- Dataset chico y didactico para clasificacion multiclase.

Decisiones tecnicas:

- Se usa division estratificada con `random_state=42`.
- Se usa `Pipeline` con `StandardScaler` y `KNeighborsClassifier(n_neighbors=5)`.
- `precision`, `recall` y `F1-score` usan promedio ponderado para resumir las tres clases.
- Se genera una matriz de confusion para `setosa`, `versicolor` y `virginica`.

### SVM para clasificacion

Estado: implementado.

Dataset:

- `Breast Cancer`.

Motivo:

- Dataset numerico adecuado para clasificacion binaria.

Decisiones tecnicas:

- Se usa la misma division estratificada de Breast Cancer que en regresion logistica.
- Se usa `Pipeline` con `StandardScaler` y `SVC(kernel="rbf")`.
- Se considera `malignant` (`0`) como clase positiva.
- Se genera una matriz de confusion para comparar directamente ambos modelos.

### Naive Bayes

Estado: implementado.

Dataset:

- `Wine`.

Motivo:

- Permite clasificacion multiclase sin entrar todavia en procesamiento de texto.

Decisiones tecnicas:

- Se usa `GaussianNB` porque Wine contiene variables numericas continuas.
- Se usa division estratificada con `random_state=42`.
- No se aplica escalado porque el modelo estima media y varianza por clase.
- Las metricas multiclase usan promedio ponderado.

### Arbol de decision para clasificacion

Estado: implementado.

Dataset:

- `Iris`.

Motivo:

- Es simple de explicar y permite entender reglas de decision.

Decisiones tecnicas:

- Se usa la misma division estratificada de Iris que en KNN.
- Se usa `DecisionTreeClassifier(max_depth=3, random_state=42)`.
- No se aplica escalado porque el arbol toma decisiones mediante cortes por variable.
- Las metricas multiclase usan promedio ponderado.

### Bosque aleatorio para clasificacion

Estado: pendiente.

Dataset propuesto:

- `Wine`.

Motivo:

- Permite comparar un ensamble contra modelos mas simples en un problema multiclase.

## Metricas

### Regresion

Metricas usadas:

- `MAE`: error absoluto medio.
- `MSE`: error cuadratico medio.
- `RMSE`: raiz del error cuadratico medio.
- `R2`: coeficiente de determinacion.

### Clasificacion

Metricas usadas:

- `accuracy`.
- `precision`.
- `recall`.
- `f1-score`.
- Matriz de confusion.

## Estado actual del menu

Opciones completas:

- `1. Regresion lineal simple`.
- `2. Regresion lineal multiple`.
- `3. Regresion polinomial`.
- `4. SVR - Regresion con vectores de soporte`.
- `5. Arbol de decision - Regresion`.
- `6. Bosque aleatorio - Regresion`.
- `7. Regresion logistica`.
- `8. K vecinos mas cercanos (KNN)`.
- `9. SVM - Clasificacion`.
- `10. Naive Bayes`.
- `11. Arbol de decision - Clasificacion`.

Opciones pendientes:

- `12. Bosque aleatorio - Clasificacion`.

## Proxima tarea recomendada

Implementar `12. Bosque aleatorio - Clasificacion`.

La implementacion deberia:

- Reutilizar el dataset `Wine`.
- Usar `RandomForestClassifier` con `random_state=42`.
- Calcular metricas multiclase con promedio ponderado.
- Generar una matriz de confusion con las tres clases.
- Guardar informe con interpretacion especifica.
- Crear informe manual en `manual_algoritmos/12_bosque_aleatorio_clasificacion.md`.
