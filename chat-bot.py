# Documentação: https://ai.google.dev/gemini-api/docs?hl=pt-br
# Instalação do SDK da Google: pip install -q -U google-generativeai
# import da biblioteca da IA generativa do Google
import google.generativeai as genai

# Utilização da biblioteca dotenv para utilizar a chave da Google API
import os
from dotenv import load_dotenv
# Carrega as variáveis do .dotenv
load_dotenv()
# Recebera a API KEY do Google Studio IA, salva no arquivo .env
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# configuração da API e criação do client do Gemini para ser utilizado
genai.configure(api_key=GOOGLE_API_KEY)

# Listar os modelos generativos disponíveis: 
# for modelSupported in genai.list_models():
    # if 'generateContent' in modelSupported.supported_generation_methods:
        # print(modelSupported.name)

# Configuração de parâmetros de criação generativa. Gemini 1.5 não suporta essas customizações
generation_config = {
    # Opções de resposta
    "candidate_count": 1,
    # Temperature: de 0 a 1. Quanto mais próximo de 1, mais criativo. Quanto mais próximo de 0, mais preciso.
    "temperature": 0.8,
}

# Configuração de segurança, que também são utilizadas no Google Studio AI
# BLOCK_NONE: bloqueia nenhum
# BLOCK_ONLY_HIGH: bloqueia poucos
# BLOCK_MEDIUM_AND_ABOVE: bloqueia albuns
# BLOCK_LOW_AND_ABOVE: bloqueia muitos
safety_settings = { 
    "HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    "HATE": "BLOCK_MEDIUM_AND_ABOVE",
    "SEXUAL": "BLOCK_MEDIUM_AND_ABOVE",
    "DANGEROUS": "BLOCK_MEDIUM_AND_ABOVE"
}

# Inicialização do modelo generativo
model = genai.GenerativeModel(
    model_name = 'gemini-pro',
    generation_config = generation_config,
    safety_settings = safety_settings
)

# Geração de conteúdo
# response = model.generate_content("O que é C++?")
# print(response.text)

# Inicialização do chat com a opção de histórico no chat
chat = model.start_chat(history=[])
# Criação da variável de input
print("Insira o prompt. \nQuando finalizar as consultas, digite Fim\n")
prompt = input("Esperando prompt: ")

while prompt != "Fim":
    # Resposta que armazenará o que foi mandado de mensagem para o chat
    response = chat.send_message(prompt)
    print("Resposta: ", response.text)
    print("--------------------------------\n")
    prompt = input("Esperando prompt: ")
