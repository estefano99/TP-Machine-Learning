"""Algoritmos de regresion del TP."""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data.loaders import (
    load_diabetes_multiple_regression,
    load_diabetes_simple_regression,
)
from utils.dataset_preview import print_dataset_preview
from utils.metrics import calculate_regression_metrics, print_regression_metrics
from utils.plots import save_real_vs_predicted_plot, save_simple_regression_plot
from utils.reports import save_algorithm_report


def run_simple_linear_regression():
    """Ejecuta regresion lineal simple con el dataset Diabetes."""
    dataset = load_diabetes_simple_regression()
    x = dataset["x"]
    y = dataset["y"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Llamar y entrenar al modelo
    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = calculate_regression_metrics(y_test, y_pred)

    print("\nResultados - Regresion lineal simple")
    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Variable de entrada: {dataset['feature_name']}")
    print(f"Variable objetivo: {dataset['target_name']}")
    print_dataset_preview(dataset)
    print_regression_metrics(metrics)

    graph_path = save_simple_regression_plot(
        x_test=x_test,
        y_test=y_test,
        y_pred=y_pred,
        feature_name=dataset["feature_name"],
        output_filename="regresion_lineal_simple_diabetes.png",
    )

    report_path = save_algorithm_report(
        algorithm_name="Regresion lineal simple",
        dataset_name=dataset["dataset_name"],
        dataset_reason=(
            "Se usa Diabetes porque esta incluido en scikit-learn, es liviano "
            "y permite practicar una prediccion numerica con una variable."
        ),
        metrics=metrics,
        graph_path=graph_path,
        output_filename="regresion_lineal_simple_diabetes.txt",
        dataset_info=dataset,
        interpretation=(
            "El modelo intenta predecir la progresion de la enfermedad "
            "a partir de una sola variable: bmi. Al usar una unica variable, "
            "el resultado sirve como primera aproximacion y no se espera "
            "un ajuste perfecto."
        ),
    )

    print(f"\nGrafico guardado en: {graph_path}")
    print(f"Informe guardado en: {report_path}")


def run_multiple_linear_regression():
    """Ejecuta regresion lineal multiple con el dataset Diabetes."""
    dataset = load_diabetes_multiple_regression()
    x = dataset["x"]
    y = dataset["y"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = calculate_regression_metrics(y_test, y_pred)

    print("\nResultados - Regresion lineal multiple")
    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Variable objetivo: {dataset['target_name']}")
    print_dataset_preview(dataset)
    print_regression_metrics(metrics)

    graph_path = save_real_vs_predicted_plot(
        y_test=y_test,
        y_pred=y_pred,
        title="Regresion lineal multiple - Diabetes",
        output_filename="regresion_lineal_multiple_diabetes.png",
    )

    report_path = save_algorithm_report(
        algorithm_name="Regresion lineal multiple",
        dataset_name=dataset["dataset_name"],
        dataset_reason=(
            "Se usa Diabetes porque incluye multiples variables numericas "
            "del paciente y permite comparar el resultado contra la regresion "
            "lineal simple usando el mismo problema."
        ),
        metrics=metrics,
        graph_path=graph_path,
        output_filename="regresion_lineal_multiple_diabetes.txt",
        dataset_info=dataset,
        interpretation=(
            "El modelo usa todas las variables disponibles del dataset para "
            "predecir la progresion de la enfermedad. Al incorporar mas "
            "informacion que la regresion lineal simple, se espera un mejor "
            "ajuste general del modelo."
        ),
    )

    print(f"\nGrafico guardado en: {graph_path}")
    print(f"Informe guardado en: {report_path}")
