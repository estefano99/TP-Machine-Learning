"""Funciones para generar graficos."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
GRAPHS_DIR = BASE_DIR / "outputs" / "graphs"


def save_simple_regression_plot(x_test, y_test, y_pred, feature_name, output_filename):
    """Guarda un grafico de puntos reales y linea de prediccion."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GRAPHS_DIR / output_filename

    plot_data = pd.DataFrame(
        {
            feature_name: x_test[feature_name].to_numpy(),
            "real": y_test.to_numpy(),
            "predicted": y_pred,
        }
    ).sort_values(feature_name)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        plot_data[feature_name],
        plot_data["real"],
        color="steelblue",
        label="Valores reales",
    )
    plt.plot(
        plot_data[feature_name],
        plot_data["predicted"],
        color="darkred",
        linewidth=2,
        label="Prediccion del modelo",
    )
    plt.title("Regresion lineal simple - Diabetes")
    plt.xlabel(feature_name)
    plt.ylabel("Progresion de la enfermedad")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def save_polynomial_regression_plot(x_test, y_test, y_pred, feature_name, output_filename):
    """Guarda un grafico de puntos reales y curva de prediccion polinomial."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GRAPHS_DIR / output_filename

    plot_data = pd.DataFrame(
        {
            feature_name: x_test[feature_name].to_numpy(),
            "real": y_test.to_numpy(),
            "predicted": y_pred,
        }
    ).sort_values(feature_name)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        plot_data[feature_name],
        plot_data["real"],
        color="steelblue",
        label="Valores reales",
    )
    plt.plot(
        plot_data[feature_name],
        plot_data["predicted"],
        color="darkred",
        linewidth=2,
        label="Curva de prediccion",
    )
    plt.title("Regresion polinomial - Diabetes")
    plt.xlabel(feature_name)
    plt.ylabel("Progresion de la enfermedad")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def save_real_vs_predicted_plot(y_test, y_pred, title, output_filename):
    """Guarda un grafico comparando valores reales contra valores predichos."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GRAPHS_DIR / output_filename

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, color="steelblue", alpha=0.8)
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        color="darkred",
        linewidth=2,
        label="Prediccion perfecta",
    )
    plt.title(title)
    plt.xlabel("Valores reales")
    plt.ylabel("Valores predichos")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def save_confusion_matrix_plot(matrix, class_names, title, output_filename):
    """Guarda un heatmap con la matriz de confusion."""
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GRAPHS_DIR / output_filename

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path
