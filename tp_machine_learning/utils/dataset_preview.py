"""Funciones para mostrar una vista previa de los datasets."""

import pandas as pd


def print_dataset_preview(dataset):
    """Muestra informacion basica y primeras filas del dataset."""
    print("\nVista previa del dataset")
    print(f"Filas: {dataset['rows']}")
    print(f"Columnas: {dataset['columns']}")
    print(f"Variable objetivo: {dataset['target_name']}")
    print(f"Variables de entrada: {', '.join(dataset['features'])}")
    print(f"\nPrimeras {dataset['preview_rows']} filas:")
    print(dataset["dataframe"].head(dataset["preview_rows"]))


def print_transformed_dataset_preview(values, feature_names, preview_rows=5):
    """Muestra una vista previa de un dataset luego de una transformacion."""
    transformed_dataframe = pd.DataFrame(values, columns=feature_names)

    print("\nVista previa del dataset transformado")
    print(f"Columnas transformadas: {', '.join(feature_names)}")
    print(f"\nPrimeras {preview_rows} filas transformadas:")
    print(transformed_dataframe.head(preview_rows))
