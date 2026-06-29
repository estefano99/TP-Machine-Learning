# TP Machine Learning - Python

Proyecto para el Trabajo Practico Integrador de la materia Programacion en Python.

El objetivo es desarrollar una aplicacion de consola que permita ejecutar distintos algoritmos de Machine Learning usando datasets diferentes a los utilizados en los ejemplos de la playlist de la materia.

## Objetivo del proyecto

La aplicacion tendra un menu interactivo desde consola. Desde ese menu se podran seleccionar algoritmos de regresion y clasificacion, entrenar modelos, ver metricas, generar graficos basicos y guardar informes con los resultados obtenidos.

La idea principal es construir una solucion clara, modular y realista para una entrega de facultad, sin sobrecomplicar la arquitectura.

## Algoritmos previstos

### Regresion

- Regresion lineal simple
- Regresion lineal multiple
- Regresion polinomial
- SVR, regresion con vectores de soporte
- Arbol de decision para regresion
- Bosque aleatorio para regresion

### Clasificacion

- Regresion logistica
- K vecinos mas cercanos, KNN
- SVM para clasificacion
- Naive Bayes
- Arbol de decision para clasificacion
- Bosque aleatorio para clasificacion

## Datasets

Se priorizaran datasets incluidos en `scikit-learn` para evitar dependencias externas y facilitar la ejecucion del proyecto.

Algunos datasets previstos:

- `Diabetes`
- `California Housing`
- `Breast Cancer`
- `Iris`
- `Wine`
- `Digits`

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
    metrics.py
    plots.py
    reports.py

  outputs/
    graphs/
    reports/
```

Responsabilidades principales:

- `main.py`: muestra el menu y conecta cada opcion con su algoritmo.
- `data/`: contiene funciones para cargar datasets.
- `models/`: contiene las funciones que entrenan y ejecutan modelos.
- `utils/metrics.py`: calcula y muestra metricas.
- `utils/plots.py`: genera graficos.
- `utils/reports.py`: guarda informes de resultados.
- `outputs/`: guarda graficos e informes generados.

## Estado actual

Ya se implemento:

- Menu de consola.
- Opcion `0` para salir.
- Validacion de opciones invalidas.
- Primer algoritmo completo: regresion lineal simple con dataset `Diabetes`.
- Segundo algoritmo completo: regresion lineal multiple con dataset `Diabetes`.
- Tercer algoritmo completo: regresion polinomial con dataset `Diabetes`.
- Cuarto algoritmo completo: SVR con una muestra reproducible de `California Housing`.
- Pipeline de `StandardScaler` y `SVR` para aplicar el escalado correctamente.
- Quinto algoritmo completo: arbol de decision para regresion con `California Housing`.
- Sexto algoritmo completo: bosque aleatorio para regresion con `California Housing`.
- Primer algoritmo de clasificacion: regresion logistica con `Breast Cancer`.
- Metricas de clasificacion y heatmap de matriz de confusion.
- Segundo algoritmo de clasificacion: KNN con el dataset `Iris`.
- Soporte para metricas y matrices de confusion multiclase.
- Tercer algoritmo de clasificacion: SVM con `Breast Cancer`.
- Cuarto algoritmo de clasificacion: Gaussian Naive Bayes con `Wine`.
- Vista previa del dataset en consola antes de entrenar cada modelo.
- Calculo de metricas de regresion:
  - MAE
  - MSE
  - RMSE
  - R2
- Generacion de grafico.
- Generacion de informe `.txt` con explicacion de metricas.

## Como ejecutar el proyecto

Instalar dependencias:

```bash
python -m pip install -r tp_machine_learning/requirements.txt
```

Ejecutar el menu:

```bash
python tp_machine_learning/main.py
```

Luego elegir una opcion del menu. Por ahora, las opciones completas son:

```text
1. Regresion lineal simple
2. Regresion lineal multiple
3. Regresion polinomial
4. SVR - Regresion con vectores de soporte
5. Arbol de decision - Regresion
6. Bosque aleatorio - Regresion
7. Regresion logistica
8. K vecinos mas cercanos (KNN)
9. SVM - Clasificacion
10. Naive Bayes
```

## Etapas de desarrollo

1. Crear menu minimo funcionando.
2. Implementar primer algoritmo completo.
3. Reutilizar funciones comunes.
4. Agregar el resto de algoritmos.
5. Mejorar graficos e informes.
6. Preparar informe final para la entrega.

## Informe final esperado

El informe final deberia incluir:

- Objetivo del trabajo.
- Explicacion breve de Machine Learning supervisado.
- Diferencia entre regresion y clasificacion.
- Algoritmos aplicados.
- Dataset elegido para cada algoritmo.
- Metricas obtenidas.
- Capturas de resultados.
- Graficos generados.
- Interpretacion de resultados.
- Conclusion personal del trabajo.

## Notas

Los archivos generados dentro de `tp_machine_learning/outputs/` no se suben al repositorio porque estan ignorados en `.gitignore`. Se pueden volver a generar ejecutando los algoritmos desde el menu.
