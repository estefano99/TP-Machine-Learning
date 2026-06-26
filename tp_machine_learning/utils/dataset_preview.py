"""Funciones para mostrar una vista previa de los datasets."""


def print_dataset_preview(dataset):
    """Muestra informacion basica y primeras filas del dataset."""
    print("\nVista previa del dataset")
    print(f"Filas: {dataset['rows']}")
    print(f"Columnas: {dataset['columns']}")
    print(f"Variable objetivo: {dataset['target_name']}")
    print(f"Variables de entrada: {', '.join(dataset['features'])}")
    print(f"\nPrimeras {dataset['preview_rows']} filas:")
    print(dataset["dataframe"].head(dataset["preview_rows"]))
