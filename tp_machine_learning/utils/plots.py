"""Funciones para generar graficos."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
