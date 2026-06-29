"""Funciones para cargar datasets del proyecto."""

from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)


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

# sample_size=5000 limita a 5000 filas para que el entrenamiento sea mas rapido, ya que SVR es costoso computacionalmente
# random_state=42 siempre la misma muestra
def load_california_housing_sample(sample_size=5000, random_state=42):
    """Carga una muestra reproducible del dataset California Housing."""
    california = fetch_california_housing(as_frame=True)
    target_name = california.target.name

    dataframe = california.frame.sample(
        n=sample_size,
        random_state=random_state,
    ).reset_index(drop=True)
    x = dataframe[california.feature_names]
    y = dataframe[target_name]

    return {
        "dataset_name": "California Housing",
        "target_name": target_name,
        "dataframe": dataframe,
        "features": list(x.columns),
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "preview_rows": 5,
        "sample_size": sample_size,
        "sample_random_state": random_state,
        "x": x,
        "y": y,
    }


def load_breast_cancer_classification():
    """Carga Breast Cancer para un problema de clasificacion binaria."""
    cancer = load_breast_cancer(as_frame=True)
    target_name = cancer.target.name
    x = cancer.data
    y = cancer.target
    dataframe = x.copy()
    dataframe[target_name] = y

    return {
        "dataset_name": "Breast Cancer Wisconsin",
        "target_name": target_name,
        "target_names": list(cancer.target_names),
        "positive_label": 0,
        "positive_class_name": cancer.target_names[0],
        "dataframe": dataframe,
        "features": list(x.columns),
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "preview_rows": 5,
        "x": x,
        "y": y,
    }


def load_iris_classification():
    """Carga Iris para un problema de clasificacion multiclase."""
    iris = load_iris(as_frame=True)
    target_name = iris.target.name
    x = iris.data
    y = iris.target
    dataframe = x.copy()
    dataframe[target_name] = y

    return {
        "dataset_name": "Iris",
        "target_name": target_name,
        "target_names": list(iris.target_names),
        "class_labels": [0, 1, 2],
        "dataframe": dataframe,
        "features": list(x.columns),
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "preview_rows": 5,
        "x": x,
        "y": y,
    }


def load_wine_classification():
    """Carga Wine para un problema de clasificacion multiclase."""
    wine = load_wine(as_frame=True)
    target_name = wine.target.name
    x = wine.data
    y = wine.target
    dataframe = x.copy()
    dataframe[target_name] = y

    return {
        "dataset_name": "Wine",
        "target_name": target_name,
        "target_names": list(wine.target_names),
        "class_labels": [0, 1, 2],
        "dataframe": dataframe,
        "features": list(x.columns),
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "preview_rows": 5,
        "x": x,
        "y": y,
    }
