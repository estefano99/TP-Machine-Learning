"""Funciones para cargar datasets del proyecto."""

from sklearn.datasets import load_diabetes


def load_diabetes_simple_regression():
    """Carga Diabetes usando una sola variable para regresion lineal simple."""
    diabetes = load_diabetes(as_frame=True)
    feature_name = "bmi"
    target_name = "disease_progression"

    x = diabetes.data[[feature_name]]
    y = diabetes.target
    dataframe = x.copy()
    dataframe[target_name] = y

    return {
        "dataset_name": "Diabetes",
        "feature_name": feature_name,
        "target_name": target_name,
        "dataframe": dataframe,
        "features": [feature_name],
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "preview_rows": 5,
        "x": x,
        "y": y,
    }


def load_diabetes_multiple_regression():
    """Carga Diabetes usando todas sus variables para regresion lineal multiple."""
    diabetes = load_diabetes(as_frame=True)
    target_name = "disease_progression"

    x = diabetes.data
    y = diabetes.target
    dataframe = x.copy()
    dataframe[target_name] = y

    return {
        "dataset_name": "Diabetes",
        "target_name": target_name,
        "dataframe": dataframe,
        "features": list(x.columns),
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "preview_rows": 5,
        "x": x,
        "y": y,
    }
