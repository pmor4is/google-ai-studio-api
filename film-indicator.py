import google.generativeai as genai

# Utilização da biblioteca dotenv para utilizar a chave da Google API
import os
from dotenv import load_dotenv
# Carrega as variáveis do .dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    # Opções de resposta
    "candidate_count": 1,
    # Temperature: de 0 a 1. Quanto mais próximo de 1, mais criativo. Quanto mais próximo de 0, mais preciso.
    "temperature": 0.8,
}
safety_settings = { 
    "HARASSMENT": "BLOCK_NONE",
    "HATE": "BLOCK_NONE",
    "SEXUAL": "BLOCK_NONE",
    "DANGEROUS": "BLOCK_NONE"
}
model = genai.GenerativeModel(
    model_name = 'gemini-pro',
    generation_config = generation_config,
    safety_settings = safety_settings,
)
chat = model.start_chat(history=[])

response = chat.send_message("(IDEIA: você analisará quatro respostas minhas, para quatro perguntas que você vai fazer para mim, de forma criativa. Somente no final das quatro perguntas você me indicará um filme ou uma série, com base nessas quatro respostas. REQUISITOS: NÃO FAÇA DUAS PERGUNTAS DE UMA VEZ, faça uma pergunta e espere a resposta e NÃO indique o filme antes das quatro perguntas. PERGUNTAS: PRIMEIRA: qual gênero você gostaria de assistir? SEGUNDA: cite três filmes desse mesmo gênero que você gostou de ver? TERCEIRA: qual tema você gostaria de assistir? QUARTA: você prefere uma série ou filme?) Faça a primeira pergunta agora")

for i in range(4):
    print("\n", response.text)
    prompt = input("\nEsperando prompt: ")
    response = chat.send_message(prompt)
    
print("\n", response.text)