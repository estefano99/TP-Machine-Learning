class Laptop:
  def __init__(self, marca, ssd, procesador, memoria):
    self.marca = marca
    self.procesador = procesador
    self.memoria = memoria
    self.ssd = ssd
    
if __name__ == '__main__':
  laptot1 = Laptop("msci", True, "I-5", "36RAM")
  
  print(laptot1.__dict__)
  print(laptot1.__delattr__)