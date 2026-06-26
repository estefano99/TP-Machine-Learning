class Personaje:
    def __init__(self, vida, posicion, velocidad, nombre="Personaje"):
        self.vida = vida
        self.posicion = posicion
        self.velocidad = velocidad
        self.nombre = nombre

    def recibir_ataque(self, cantidad):
        self.vida -= cantidad
        print(f"{self.nombre} recibio {cantidad} de daño. Vida actual: {self.vida}")

        if self.vida <= 0:
            print(f"{self.nombre} ha muerto.")

    def mover(self, direccion):
        if direccion == "derecha":
            self.posicion += self.velocidad
        elif direccion == "izquierda":
            self.posicion -= self.velocidad
        else:
            print("Direccion invalida. Use 'derecha' o 'izquierda'.")
            return

        print(f"{self.nombre} se movio hacia {direccion}. Posicion actual: {self.posicion}")


class Soldado(Personaje):
    def __init__(self, vida, posicion, velocidad, ataque, nombre="Soldado"):
        super().__init__(vida, posicion, velocidad, nombre)
        self.ataque = ataque

    def atacar(self, otro_personaje):
        if isinstance(otro_personaje, (Soldado, Campesino)):
            print(f"{self.nombre} ataca a {otro_personaje.nombre}.")
            otro_personaje.recibir_ataque(self.ataque)
        else:
            print("El objetivo no es un personaje valido.")


class Campesino(Personaje):
    def __init__(self, vida, posicion, velocidad, cosecha, nombre="Campesino"):
        super().__init__(vida, posicion, velocidad, nombre)
        self.cosecha = cosecha

    def atacar(self, otro_personaje):
        print(f"{self.nombre} no puede atacar.")

    def cosechar(self):
        print(f"{self.nombre} cosecho {self.cosecha} unidades.")
        return self.cosecha


if __name__ == "__main__":
    soldado_1 = Soldado(vida=100, posicion=0, velocidad=5, ataque=30, nombre="Soldado 1")
    soldado_2 = Soldado(vida=60, posicion=10, velocidad=3, ataque=20, nombre="Soldado 2")
    campesino_1 = Campesino(vida=50, posicion=20, velocidad=2, cosecha=15, nombre="Campesino 1")
    campesino_2 = Campesino(vida=40, posicion=30, velocidad=1, cosecha=10, nombre="Campesino 2")

    print("Movimiento de personajes")
    soldado_1.mover("derecha")
    campesino_1.mover("izquierda")
    campesino_2.mover("arriba")

    print("\nAtaques permitidos")
    soldado_1.atacar(soldado_2)
    soldado_1.atacar(campesino_1)

    print("\nAtaque hasta dejar vida en 0 o menos")
    soldado_1.atacar(campesino_1)

    print("\nCosecha")
    cantidad_cosechada = campesino_2.cosechar()
    print(f"Cantidad devuelta por cosechar: {cantidad_cosechada}")

    print("\nRestriccion de campesinos")
    campesino_1.atacar(soldado_1)
    campesino_1.atacar(campesino_2)
