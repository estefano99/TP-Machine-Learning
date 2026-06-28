"""Algoritmos de regresion del TP."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from data.loaders import (
    load_california_housing_sample,
    load_diabetes_multiple_regression,
    load_diabetes_simple_regression,
)
from utils.dataset_preview import print_dataset_preview, print_transformed_dataset_preview
from utils.metrics import calculate_regression_metrics, print_regression_metrics
from utils.plots import (
    save_polynomial_regression_plot,
    save_real_vs_predicted_plot,
    save_simple_regression_plot,
)
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


def run_polynomial_regression():
    """Ejecuta regresion polinomial con el dataset Diabetes."""
    dataset = load_diabetes_simple_regression()
    x = dataset["x"]
    y = dataset["y"]
    degree = 2

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    x_train_poly = polynomial_features.fit_transform(x_train)
    x_test_poly = polynomial_features.transform(x_test)
    polynomial_feature_names = polynomial_features.get_feature_names_out(
        [dataset["feature_name"]]
    )

    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    y_pred = model.predict(x_test_poly)
    metrics = calculate_regression_metrics(y_test, y_pred)

    print("\nResultados - Regresion polinomial")
    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Variable de entrada: {dataset['feature_name']}")
    print(f"Variable objetivo: {dataset['target_name']}")
    print(f"Grado del polinomio: {degree}")
    print_dataset_preview(dataset)
    print_transformed_dataset_preview(
        values=polynomial_features.transform(x),
        feature_names=polynomial_feature_names,
        preview_rows=dataset["preview_rows"],
    )
    print_regression_metrics(metrics)

    graph_path = save_polynomial_regression_plot(
        x_test=x_test,
        y_test=y_test,
        y_pred=y_pred,
        feature_name=dataset["feature_name"],
        output_filename="regresion_polinomial_diabetes.png",
    )

    report_path = save_algorithm_report(
        algorithm_name="Regresion polinomial",
        dataset_name=dataset["dataset_name"],
        dataset_reason=(
            "Se usa Diabetes con la variable bmi para comparar este modelo "
            "contra la regresion lineal simple usando el mismo dataset y la "
            "misma variable de entrada."
        ),
        metrics=metrics,
        graph_path=graph_path,
        output_filename="regresion_polinomial_diabetes.txt",
        dataset_info=dataset,
        transformation_info=(
            "Se aplico PolynomialFeatures de grado 2 sobre la variable bmi. "
            "El modelo entrena con las columnas transformadas: "
            f"{', '.join(polynomial_feature_names)}."
        ),
        interpretation=(
            "El modelo transforma la variable bmi en caracteristicas "
            "polinomiales de grado 2 para intentar capturar una relacion "
            "curva entre la variable de entrada y la progresion de la "
            "enfermedad. Como sigue usando una sola variable original, puede "
            "mejorar la forma del ajuste, pero sigue teniendo informacion "
            "limitada."
        ),
    )

    print(f"\nGrafico guardado en: {graph_path}")
    print(f"Informe guardado en: {report_path}")


def run_svr_regression():
    """Ejecuta SVR con una muestra del dataset California Housing."""
    dataset = load_california_housing_sample(sample_size=5000, random_state=42)
    x = dataset["x"]
    y = dataset["y"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

# pipeline hace que el StandardScaler se aplique antes de entrenar y predecir con SVR, para evitar que las variables con valores numericos mayores dominen el modelo. Esto es importante porque SVR es sensible a la escala de las variables.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf")),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    metrics = calculate_regression_metrics(y_test, y_pred)

    print("\nResultados - SVR")
    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Variable objetivo: {dataset['target_name']}")
    print(
        "Muestra: "
        f"{dataset['sample_size']} registros "
        f"(random_state={dataset['sample_random_state']})"
    )
    print("Pipeline: StandardScaler -> SVR (kernel rbf)")
    print_dataset_preview(dataset)
    print_regression_metrics(metrics)

    graph_path = save_real_vs_predicted_plot(
        y_test=y_test,
        y_pred=y_pred,
        title="SVR - California Housing",
        output_filename="svr_california_housing.png",
    )

    report_path = save_algorithm_report(
        algorithm_name="SVR - Regresion con vectores de soporte",
        dataset_name=dataset["dataset_name"],
        dataset_reason=(
            "Se usa California Housing porque contiene multiples variables "
            "numericas relacionadas con el valor de las viviendas y permite "
            "aplicar SVR a un problema de regresion real."
        ),
        metrics=metrics,
        graph_path=graph_path,
        output_filename="svr_california_housing.txt",
        dataset_info=dataset,
        transformation_info=(
            "Se usa una muestra fija de 5.000 registros para reducir el "
            "tiempo de entrenamiento y mantener reproducibilidad. La muestra "
            "se obtiene con random_state=42. StandardScaler y SVR con kernel "
            "rbf se integran en un Pipeline para aplicar el mismo escalado "
            "durante el entrenamiento y la prediccion."
        ),
        interpretation=(
            "El modelo usa todas las variables de la muestra para predecir "
            "el valor medio de las viviendas. SVR con kernel rbf puede "
            "representar relaciones no lineales, y el escalado evita que "
            "las variables con valores numericos mayores dominen el modelo."
        ),
    )

    print(f"\nGrafico guardado en: {graph_path}")
    print(f"Informe guardado en: {report_path}")


def run_decision_tree_regression():
    """Ejecuta un arbol de decision con California Housing."""
    dataset = load_california_housing_sample(sample_size=5000, random_state=42)
    x = dataset["x"]
    y = dataset["y"]
    max_depth = 5

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = calculate_regression_metrics(y_test, y_pred)

    print("\nResultados - Arbol de decision para regresion")
    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Variable objetivo: {dataset['target_name']}")
    print(
        "Muestra: "
        f"{dataset['sample_size']} registros "
        f"(random_state={dataset['sample_random_state']})"
    )
    print(f"Profundidad maxima del arbol: {max_depth}")
    print_dataset_preview(dataset)
    print_regression_metrics(metrics)

    graph_path = save_real_vs_predicted_plot(
        y_test=y_test,
        y_pred=y_pred,
        title="Arbol de decision - California Housing",
        output_filename="arbol_decision_regresion_california_housing.png",
    )

    report_path = save_algorithm_report(
        algorithm_name="Arbol de decision para regresion",
        dataset_name=dataset["dataset_name"],
        dataset_reason=(
            "Se usa la misma muestra de California Housing que en SVR para "
            "comparar ambos algoritmos sobre los mismos datos de entrada y "
            "la misma variable objetivo."
        ),
        metrics=metrics,
        graph_path=graph_path,
        output_filename="arbol_decision_regresion_california_housing.txt",
        dataset_info=dataset,
        transformation_info=(
            "Se usa una muestra fija de 5.000 registros con random_state=42. "
            "El arbol se limita a max_depth=5 para reducir el sobreajuste. "
            "No se aplica escalado porque los arboles toman decisiones "
            "mediante cortes sobre cada variable individual."
        ),
        interpretation=(
            "El arbol divide los datos en grupos mediante reglas sucesivas "
            "y asigna una prediccion numerica a cada grupo final. Limitar su "
            "profundidad evita que memorice demasiados detalles de los datos "
            "de entrenamiento, aunque tambien puede restringir su precision."
        ),
    )

    print(f"\nGrafico guardado en: {graph_path}")
    print(f"Informe guardado en: {report_path}")


def run_random_forest_regression():
    """Ejecuta un bosque aleatorio con California Housing."""
    dataset = load_california_housing_sample(sample_size=5000, random_state=42)
    x = dataset["x"]
    y = dataset["y"]
    number_of_trees = 100

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=number_of_trees,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = calculate_regression_metrics(y_test, y_pred)

    print("\nResultados - Bosque aleatorio para regresion")
    print(f"Dataset: {dataset['dataset_name']}")
    print(f"Variable objetivo: {dataset['target_name']}")
    print(
        "Muestra: "
        f"{dataset['sample_size']} registros "
        f"(random_state={dataset['sample_random_state']})"
    )
    print(f"Cantidad de arboles: {number_of_trees}")
    print_dataset_preview(dataset)
    print_regression_metrics(metrics)

    graph_path = save_real_vs_predicted_plot(
        y_test=y_test,
        y_pred=y_pred,
        title="Bosque aleatorio - California Housing",
        output_filename="bosque_aleatorio_regresion_california_housing.png",
    )

    report_path = save_algorithm_report(
        algorithm_name="Bosque aleatorio para regresion",
        dataset_name=dataset["dataset_name"],
        dataset_reason=(
            "Se usa la misma muestra de California Housing que en SVR y el "
            "arbol de decision para comparar los tres algoritmos sobre los "
            "mismos datos."
        ),
        metrics=metrics,
        graph_path=graph_path,
        output_filename="bosque_aleatorio_regresion_california_housing.txt",
        dataset_info=dataset,
        transformation_info=(
            "Se usa una muestra fija de 5.000 registros con random_state=42. "
            "El modelo combina 100 arboles y usa random_state=42 para "
            "mantener resultados reproducibles. No se aplica escalado porque "
            "los arboles toman decisiones mediante cortes sobre cada variable."
        ),
        interpretation=(
            "El bosque aleatorio entrena varios arboles con diferentes "
            "muestras y subconjuntos de variables. La prediccion final es el "
            "promedio de sus resultados, lo que normalmente reduce el "
            "sobreajuste y la variabilidad de un unico arbol de decision."
        ),
    )

    print(f"\nGrafico guardado en: {graph_path}")
    print(f"Informe guardado en: {report_path}")
