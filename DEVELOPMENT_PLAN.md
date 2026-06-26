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

Estado: iniciada.

Ya existen funciones comunes para:

- Vista previa de datasets: `utils/dataset_preview.py`.
- Metricas de regresion: `utils/metrics.py`.
- Graficos: `utils/plots.py`.
- Informes: `utils/reports.py`.

Pendiente:

- Agregar metricas comunes de clasificacion.
- Agregar graficos comunes de clasificacion.
- Crear `models/classification.py`.

### Etapa 4: Agregar el resto de algoritmos

Estado: en progreso.

Algoritmos de regresion:

- [x] Regresion lineal simple.
- [x] Regresion lineal multiple.
- [ ] Regresion polinomial.
- [ ] SVR, regresion con vectores de soporte.
- [ ] Arbol de decision para regresion.
- [ ] Bosque aleatorio para regresion.

Algoritmos de clasificacion:

- [ ] Regresion logistica.
- [ ] K vecinos mas cercanos, KNN.
- [ ] SVM para clasificacion.
- [ ] Naive Bayes.
- [ ] Arbol de decision para clasificacion.
- [ ] Bosque aleatorio para clasificacion.

### Etapa 5: Graficos e informes

Estado: iniciada.

Ya implementado:

- Grafico para regresion lineal simple.
- Grafico de valores reales vs valores predichos para regresion multiple.
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

Pendiente:

- Matriz de confusion para clasificacion.
- Heatmap de matriz de confusion.
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

Estado: pendiente.

Dataset propuesto:

- `Diabetes`, usando `bmi`.

Motivo:

- Permite comparar una curva polinomial contra la regresion lineal simple usando el mismo dataset.

### SVR

Estado: pendiente.

Dataset propuesto:

- `Diabetes`.

Motivo:

- Es liviano y evita tiempos de entrenamiento largos.

### Arbol de decision para regresion

Estado: pendiente.

Dataset propuesto:

- `Diabetes` o `California Housing`.

Decision sugerida:

- Usar `Diabetes` si queremos mantener todo rapido y simple.
- Usar `California Housing` si queremos mas variedad de dataset.

### Bosque aleatorio para regresion

Estado: pendiente.

Dataset propuesto:

- `Diabetes` o `California Housing`.

Decision sugerida:

- Usar el mismo dataset que el arbol de decision para poder comparar resultados.

### Regresion logistica

Estado: pendiente.

Dataset propuesto:

- `Breast Cancer`.

Motivo:

- Es un problema de clasificacion binaria claro y viene incluido en `scikit-learn`.

### KNN

Estado: pendiente.

Dataset propuesto:

- `Iris`.

Motivo:

- Dataset chico y didactico para clasificacion multiclase.

### SVM para clasificacion

Estado: pendiente.

Dataset propuesto:

- `Breast Cancer`.

Motivo:

- Dataset numerico adecuado para clasificacion binaria.

### Naive Bayes

Estado: pendiente.

Dataset propuesto:

- `Wine`.

Motivo:

- Permite clasificacion multiclase sin entrar todavia en procesamiento de texto.

### Arbol de decision para clasificacion

Estado: pendiente.

Dataset propuesto:

- `Iris`.

Motivo:

- Es simple de explicar y permite entender reglas de decision.

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

Metricas pendientes de implementar:

- `accuracy`.
- `precision`.
- `recall`.
- `f1-score`.
- Matriz de confusion.

## Estado actual del menu

Opciones completas:

- `1. Regresion lineal simple`.
- `2. Regresion lineal multiple`.

Opciones pendientes:

- `3` a `12`.

## Proxima tarea recomendada

Implementar `3. Regresion polinomial` usando `Diabetes` con la variable `bmi`.

La implementacion deberia:

- Reutilizar el loader simple de Diabetes o crear uno especifico si hace falta.
- Usar `PolynomialFeatures`.
- Usar `LinearRegression`.
- Calcular las mismas metricas de regresion.
- Generar grafico de puntos reales y curva polinomial.
- Guardar informe con interpretacion especifica.
