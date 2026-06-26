"""Algoritmos de regresion del TP."""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data.loaders import load_diabetes_simple_regression
from utils.metrics import calculate_regression_metrics, print_regression_metrics
from utils.plots import save_simple_regression_plot
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
    )

    print(f"\nGrafico guardado en: {graph_path}")
    print(f"Informe guardado en: {report_path}")
