print("hola mundo")

def saludar(nombre):
    print(f"Hola {nombre}")

saludar("Alice")

nota_alumno = input("Ingrese la nota del alumno: ")

def evaluar_nota(nota):
  if nota >= 7:
    print("Aprobado")
  else:
    print("Reprobado")
    
evaluar_nota(float(nota_alumno))