# Biblioteca para realizar requisições
import requests
# Biblioteca para pegar algo aleatório
import random

url = 'https://raw.githubusercontent.com/guilhermeonrails/api-imersao-ia/main/words.json'
# Requisição
response = requests.get(url)
# Transforma em JSON
data = response.json()

# Utilizando a biblioteca aleatória
secret_value = random.choice(data)

secret_word = secret_value['palavra']
tip = secret_value['dica']

print(f'A palavra secreta possui {len(secret_word)} letras. -> Dica: {tip}')


guess = input('O palpite é: ')
if guess == secret_word:
    print('Acertou')
else: 
    print(f'Errou. Resposta correta: {secret_word}')