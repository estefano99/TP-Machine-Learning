"""Menu principal del TP de Machine Learning."""

from models.regression import (
    run_multiple_linear_regression,
    run_polynomial_regression,
    run_simple_linear_regression,
    run_svr_regression,
)


MENU_OPTIONS = {
    "1": "Regresion lineal simple",
    "2": "Regresion lineal multiple",
    "3": "Regresion polinomial",
    "4": "SVR - Regresion con vectores de soporte",
    "5": "Arbol de decision - Regresion",
    "6": "Bosque aleatorio - Regresion",
    "7": "Regresion logistica",
    "8": "K vecinos mas cercanos (KNN)",
    "9": "SVM - Clasificacion",
    "10": "Naive Bayes",
    "11": "Arbol de decision - Clasificacion",
    "12": "Bosque aleatorio - Clasificacion",
}


def show_menu():
    """Muestra las opciones disponibles en consola."""
    print("\n=== TP Machine Learning ===")
    print("\nRegresion")
    print("1. Regresion lineal simple")
    print("2. Regresion lineal multiple")
    print("3. Regresion polinomial")
    print("4. SVR - Regresion con vectores de soporte")
    print("5. Arbol de decision - Regresion")
    print("6. Bosque aleatorio - Regresion")

    print("\nClasificacion")
    print("7. Regresion logistica")
    print("8. K vecinos mas cercanos (KNN)")
    print("9. SVM - Clasificacion")
    print("10. Naive Bayes")
    print("11. Arbol de decision - Clasificacion")
    print("12. Bosque aleatorio - Clasificacion")

    print("\n0. Salir")


def handle_option(option):
    """Procesa una opcion valida del menu."""
    algorithm_name = MENU_OPTIONS[option]
    print(f"\nSeleccionaste: {algorithm_name}")

    if option == "1":
        run_simple_linear_regression()
    elif option == "2":
        run_multiple_linear_regression()
    elif option == "3":
        run_polynomial_regression()
    elif option == "4":
        run_svr_regression()
    else:
        print("Funcion en desarrollo.")


def main():
    """Ejecuta el menu interactivo hasta que el usuario elija salir."""
    while True:
        show_menu()
        option = input("\nIngrese una opcion: ").strip()

        if option == "0":
            print("\nPrograma finalizado.")
            break

        if option in MENU_OPTIONS:
            handle_option(option)
        else:
            print("\nOpcion invalida. Intente nuevamente.")


if __name__ == "__main__":
    main()
