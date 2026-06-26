# Plantilla de Informe por Algoritmo

Este archivo define el formato que se debe usar cada vez que se implemente un algoritmo nuevo en el TP.

Objetivo: generar un informe en Markdown que sirva como material de estudio y documentacion del manual del TP, no solo como resumen tecnico del codigo.

## Ubicacion sugerida de los informes

Los informes manuales de estudio se deben guardar en:

```text
manual_algoritmos/
```

Nombre sugerido:

```text
01_regresion_lineal_simple.md
02_regresion_lineal_multiple.md
03_regresion_polinomial.md
```

## Regla de trabajo

Cada vez que se implemente un algoritmo nuevo, tambien se debe crear o actualizar su informe Markdown siguiendo esta plantilla.

El informe debe basarse en:

- Codigo implementado.
- Dataset utilizado.
- Metricas reales obtenidas en la ultima ejecucion.
- Grafico generado.
- Interpretacion academica del resultado.

## Formato del informe

# Informe - Nombre del algoritmo

## 1. Nombre del algoritmo

Indicar claramente el nombre del algoritmo implementado.

Ejemplo:

- Regresion lineal multiple.

## 2. Objetivo del algoritmo

Explicar brevemente:

- que problema resuelve;
- si es regresion o clasificacion;
- que intenta predecir;
- que diferencia tiene con el algoritmo anterior, si corresponde.

## 3. Dataset utilizado

Indicar:

- nombre del dataset;
- libreria o fuente;
- cantidad de filas y columnas;
- variable objetivo `y`;
- variables independientes `X`;
- si se usa una sola variable o multiples variables.

## 4. Archivos modificados o creados

Listar los archivos que intervienen y explicar la responsabilidad de cada uno.

Archivos habituales:

- `main.py`
- `data/loaders.py`
- `models/regression.py`
- `models/classification.py`
- `utils/dataset_preview.py`
- `utils/metrics.py`
- `utils/plots.py`
- `utils/reports.py`

Para cada archivo explicar:

- que funcion nueva se agrego;
- que recibe;
- que devuelve;
- que responsabilidad cumple dentro del flujo.

## 5. Flujo de ejecucion

Explicar paso a paso que ocurre cuando el usuario selecciona la opcion del menu:

1. Se muestra el menu.
2. El usuario elige la opcion.
3. Se carga el dataset.
4. Se separan `X` e `y`.
5. Se divide en entrenamiento y prueba.
6. Se entrena el modelo.
7. Se hacen predicciones.
8. Se calculan metricas.
9. Se genera grafico.
10. Se guarda informe.

## 6. Metricas obtenidas

Incluir las metricas reales obtenidas en la ultima ejecucion.

Para regresion:

- `MAE`
- `MSE`
- `RMSE`
- `R2`

Para clasificacion:

- `accuracy`
- `precision`
- `recall`
- `f1-score`
- matriz de confusion

Para cada metrica explicar:

- que mide;
- si cuanto mas bajo o mas alto es mejor;
- como interpretar el valor obtenido.

## 7. Grafico generado

Explicar:

- nombre del archivo generado;
- que representa el eje X;
- que representa el eje Y;
- que representa cada punto;
- que representa la linea roja o linea de referencia;
- como interpretar puntos cerca o lejos de la linea.

Si el grafico es de valores reales vs predichos, aclarar:

- eje X = valor real;
- eje Y = valor predicho;
- cada punto combina ambos valores para un mismo caso del dataset;
- la linea roja representa una prediccion perfecta.

## 8. Interpretacion general del resultado

Redactar una interpretacion sencilla y academica.

Debe incluir:

- si el modelo mejoro respecto al anterior;
- por que mejoro o por que no;
- que limitaciones tiene;
- que conclusion se puede sacar.

## 9. Relacion con programacion tradicional

Relacionar el modelo con conceptos de programacion:

- `X` como input;
- `y` como output esperado;
- `fit()` como entrenamiento o configuracion del modelo;
- `predict()` como metodo que devuelve una salida;
- el modelo como objeto que aprende parametros.

## 10. Relacion con econometria o estadistica

Si aplica, explicar:

- variable dependiente;
- variables independientes;
- coeficientes;
- interpretacion de `R2`;
- por que agregar mas variables puede mejorar el modelo.

## 11. Preguntas tipicas de parcial

Generar preguntas con respuestas breves.

Ejemplos:

- Que diferencia hay entre regresion lineal simple y multiple?
- Que representa `X_train`?
- Que representa `y_test`?
- Que hace `fit()`?
- Que hace `predict()`?
- Que significa `R2`?
- Por que el grafico de regresion multiple no muestra una recta tradicional?

## 12. Errores comunes o confusiones

Incluir aclaraciones sobre posibles errores conceptuales.

Ejemplos:

- creer que la linea roja del grafico siempre es la recta de regresion;
- pensar que hay puntos reales y puntos predichos separados, cuando cada punto puede combinar ambos valores;
- evaluar el modelo con los mismos datos con los que se entreno;
- confundir regresion con clasificacion.

## 13. Cosas que aprendi

Agregar una seccion escrita en primera persona, sencilla y clara, como apunte de estudio.

Ejemplo:

> Aprendi que en un grafico de valores reales vs predichos cada punto representa un caso del dataset. El eje X muestra el valor real y el eje Y muestra el valor predicho. La linea roja indica como seria una prediccion perfecta.
