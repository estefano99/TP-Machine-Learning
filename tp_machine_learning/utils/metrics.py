"""Funciones auxiliares para calcular y mostrar metricas."""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def calculate_regression_metrics(y_true, y_pred):
    """Calcula las metricas principales para evaluar un modelo de regresion."""
    mse = mean_squared_error(y_true, y_pred)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": mse ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def print_regression_metrics(metrics):
    """Muestra en consola las metricas de regresion calculadas."""
    print("\nMetricas:")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2: {metrics['R2']:.4f}")


def calculate_classification_metrics(
    y_true,
    y_pred,
    average="binary",
    positive_label=1,
    labels=None,
):
    """Calcula metricas para clasificacion binaria o multiclase."""
    score_options = {"average": average, "zero_division": 0}
    if average == "binary":
        score_options["pos_label"] = positive_label

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, **score_options),
        "Recall": recall_score(y_true, y_pred, **score_options),
        "F1-score": f1_score(y_true, y_pred, **score_options),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return metrics, matrix


def print_classification_metrics(metrics, matrix, class_names):
    """Muestra metricas y matriz de confusion de clasificacion."""
    print("\nMetricas:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nMatriz de confusion")
    class_order = ", ".join(
        f"{class_index}={class_name}"
        for class_index, class_name in enumerate(class_names)
    )
    print(f"Orden de clases: {class_order}")
    print(matrix)
