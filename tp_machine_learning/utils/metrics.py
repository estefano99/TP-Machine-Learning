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


def calculate_classification_metrics(y_true, y_pred, positive_label=1):
    """Calcula metricas para una clasificacion binaria."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=positive_label),
        "Recall": recall_score(y_true, y_pred, pos_label=positive_label),
        "F1-score": f1_score(y_true, y_pred, pos_label=positive_label),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return metrics, matrix


def print_classification_metrics(metrics, matrix, class_names):
    """Muestra metricas y matriz de confusion de clasificacion."""
    print("\nMetricas:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nMatriz de confusion")
    print(f"Orden de clases: 0={class_names[0]}, 1={class_names[1]}")
    print(matrix)
