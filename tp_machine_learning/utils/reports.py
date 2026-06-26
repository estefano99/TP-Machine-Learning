"""Funciones para guardar informes de resultados."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

METRIC_DESCRIPTIONS = {
    "MAE": "Error absoluto medio. Indica el error promedio del modelo sin considerar el signo.",
    "MSE": "Error cuadratico medio. Penaliza mas los errores grandes porque eleva los errores al cuadrado.",
    "RMSE": "Raiz del error cuadratico medio. Mide el error promedio en una escala similar a la variable objetivo.",
    "R2": "Coeficiente de determinacion. Indica que tan bien el modelo explica los datos; mientras mas cerca de 1, mejor.",
}


def save_algorithm_report(
    algorithm_name,
    dataset_name,
    dataset_reason,
    metrics,
    graph_path,
    output_filename,
    dataset_info=None,
    interpretation=None,
):
    """Guarda un informe simple en texto plano."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / output_filename

    lines = [
        f"Algoritmo: {algorithm_name}",
        f"Dataset: {dataset_name}",
        "",
        "Motivo de eleccion del dataset:",
        dataset_reason,
        "",
    ]

    if dataset_info:
        lines.extend(
            [
                "Datos utilizados:",
                f"- Filas: {dataset_info['rows']}",
                f"- Columnas: {dataset_info['columns']}",
                f"- Variables de entrada: {', '.join(dataset_info['features'])}",
                f"- Variable objetivo: {dataset_info['target_name']}",
                "",
            ]
        )

    lines.append("Metricas obtenidas:")

    for metric_name, metric_value in metrics.items():
        description = METRIC_DESCRIPTIONS.get(metric_name, "Metrica de evaluacion del modelo.")
        lines.append(f"- {metric_name}: {metric_value:.4f}")
        lines.append(f"  Significado: {description}")

    lines.extend(
        [
            "",
            f"Grafico generado: {graph_path}",
            "",
            "Interpretacion breve:",
            interpretation
            or (
                "El modelo intenta predecir la progresion de la enfermedad "
                "a partir de una sola variable. Al usar una unica variable, "
                "el resultado sirve como primera aproximacion y no se espera "
                "un ajuste perfecto."
            ),
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
