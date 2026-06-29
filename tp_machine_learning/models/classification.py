"""Algoritmos de clasificacion del TP."""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.loaders import load_breast_cancer_classification, load_iris_classification
from utils.dataset_preview import print_dataset_preview
from utils.metrics import (
    calculate_classification_metrics,
    print_classification_metrics,
)
from utils.plots import save_confusion_matrix_plot
from utils.reports import save_algorithm_report


def run_logistic_regression():
    """Ejecuta regresion logistica con el dataset Breast Cancer."""
    dataset = load_breast_cancer_classification()
    x = dataset["x"]
    y = dataset["y"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    metrics, matrix = calculate_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        positive_label=dataset["positive_label"],
    )

    print("\nResultados - Regresion logistica")
    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Variable objetivo: {dataset['target_name']}")
    print(f"Clases: {', '.join(dataset['target_names'])}")
    print(f"Clase positiva para las metricas: {dataset['positive_class_name']} (0)")
    print("Pipeline: StandardScaler -> LogisticRegression")
    print_dataset_preview(dataset)
    print_classification_metrics(metrics, matrix, dataset["target_names"])

    graph_path = save_confusion_matrix_plot(
        matrix=matrix,
        class_names=dataset["target_names"],
        title="Regresion logistica - Breast Cancer",
        output_filename="regresion_logistica_breast_cancer.png",
    )

    report_path = save_algorithm_report(
        algorithm_name="Regresion logistica",
        dataset_name=dataset["dataset_name"],
        dataset_reason=(
            "Se usa Breast Cancer porque es un problema de clasificacion "
            "binaria claro, contiene variables numericas y esta incluido en "
            "scikit-learn."
        ),
        metrics=metrics,
        graph_path=graph_path,
        output_filename="regresion_logistica_breast_cancer.txt",
        dataset_info=dataset,
        transformation_info=(
            "Se usa una division estratificada con random_state=42 para "
            "conservar la proporcion de clases. StandardScaler y "
            "LogisticRegression se integran en un Pipeline. Para precision, "
            "recall y F1-score se considera malignant (0) como clase positiva. "
            f"Matriz de confusion: {matrix.tolist()}."
        ),
        interpretation=(
            "El modelo clasifica cada tumor como maligno o benigno. En este "
            "problema el recall de la clase maligna es especialmente "
            "importante, porque indica que proporcion de los casos malignos "
            "reales fue detectada por el modelo."
        ),
    )

    print(f"\nGrafico guardado en: {graph_path}")
    print(f"Informe guardado en: {report_path}")


def run_knn_classification():
    """Ejecuta KNN con el dataset Iris."""
    dataset = load_iris_classification()
    x = dataset["x"]
    y = dataset["y"]
    number_of_neighbors = 5

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=number_of_neighbors)),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    metrics, matrix = calculate_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        average="weighted",
        labels=dataset["class_labels"],
    )

    print("\nResultados - K vecinos mas cercanos (KNN)")
    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Variable objetivo: {dataset['target_name']}")
    print(f"Clases: {', '.join(dataset['target_names'])}")
    print(f"Cantidad de vecinos: {number_of_neighbors}")
    print("Pipeline: StandardScaler -> KNeighborsClassifier")
    print("Promedio de metricas multiclase: weighted")
    print_dataset_preview(dataset)
    print_classification_metrics(metrics, matrix, dataset["target_names"])

    graph_path = save_confusion_matrix_plot(
        matrix=matrix,
        class_names=dataset["target_names"],
        title="KNN - Iris",
        output_filename="knn_iris.png",
    )

    report_path = save_algorithm_report(
        algorithm_name="K vecinos mas cercanos (KNN)",
        dataset_name=dataset["dataset_name"],
        dataset_reason=(
            "Se usa Iris porque es un dataset pequeño, equilibrado y "
            "didactico que permite practicar clasificacion de tres clases."
        ),
        metrics=metrics,
        graph_path=graph_path,
        output_filename="knn_iris.txt",
        dataset_info=dataset,
        transformation_info=(
            "Se usa una division estratificada con random_state=42. "
            "StandardScaler y KNeighborsClassifier con k=5 se integran en "
            "un Pipeline. Precision, recall y F1-score usan promedio "
            f"ponderado. Matriz de confusion: {matrix.tolist()}."
        ),
        interpretation=(
            "KNN clasifica cada flor mediante la votacion de sus cinco "
            "vecinos mas cercanos. El escalado evita que una medida con "
            "valores mayores domine el calculo de las distancias."
        ),
    )

    print(f"\nGrafico guardado en: {graph_path}")
    print(f"Informe guardado en: {report_path}")
