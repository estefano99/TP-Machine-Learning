# Incorrecto

class User:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
    
    def save_to_database(self):
        # Lógica para salvar o usuário no banco de dados
        print(f"Salvando {self.name} no banco de dados.")

    def send_email(self, message: str):
        # Lógica para enviar um email para o usuário
        print(f"Enviando email para {self.email}: {message}")

# Correto
class User:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

class UserRepository:
    def save(self, user: User):
        # Lógica para salvar o usuário no banco de dados
        print(f"Salvando {user.name} no banco de dados.")

class EmailService:
    def send_email(self, user: User, message: str):
        # Lógica para enviar um email para o usuário
        print(f"Enviando email para {user.email}: {message}")