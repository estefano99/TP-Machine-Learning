"""Funciones para cargar datasets del proyecto."""

from sklearn.datasets import load_diabetes


def load_diabetes_simple_regression():
    """Carga Diabetes usando una sola variable para regresion lineal simple."""
    diabetes = load_diabetes(as_frame=True)
    feature_name = "bmi"
    target_name = "disease_progression"

    x = diabetes.data[[feature_name]]
    y = diabetes.target

    return {
        "dataset_name": "Diabetes",
        "feature_name": feature_name,
        "target_name": target_name,
        "x": x,
        "y": y,
    }
