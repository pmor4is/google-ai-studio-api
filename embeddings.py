# Sistema para busca em documentos utilizando embeddings e Gemini API
import google.generativeai as genai
import numpy as np
import pandas as pd

# Utilização da biblioteca dotenv para utilizar a chave da Google API
import os
from dotenv import load_dotenv
# Carrega as variáveis do .dotenv
load_dotenv()
# Recebera a API KEY do Google Studio IA, salva no arquivo .env
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

# Listar os modelos de embedding disponíveis
# for modelSupported in genai.list_models():
    # if 'embedContent' in modelSupported.supported_generation_methods:
        # print(modelSupported.name )

model = "models/embedding-001"

# Exemplo de embedding
# title = "A próxima geração de IA para desenvolvedores e Google Workspace"
# sample_text = ("Título: A próxima geração de IA para desenvolvedores e Google Workspace"
#     "\n"
#     "Artigo completo:\n"
#     "\n"
#     "Gemini API & Google AI Studio: Uma maneira acessível de explorar e criar protótipos com aplicações de IA generativa"
# )
# Chamada de embedding para title e sample_text acima
# embeddings = genai.embed_content(
#     model=model,
#     content=sample_text,
#     title=title,
#     # Qual tipo de cenário o Gemini está trabalhando
#     # Vai receber documentos
#     task_type="RETRIEVAL_DOCUMENT"
# )

# Listagem de documentos que serão buscados
DOCUMENT1 = {
    "Título": "Operação do sistema de controle climático",
    "Conteúdo": "O Googlecar tem um sistema de controle climático que permite ajustar a temperatura e o fluxo de ar no carro. Para operar o sistema de controle climático, use os botões e botões localizados no console central.  Temperatura: O botão de temperatura controla a temperatura dentro do carro. Gire o botão no sentido horário para aumentar a temperatura ou no sentido anti-horário para diminuir a temperatura. Fluxo de ar: O botão de fluxo de ar controla a quantidade de fluxo de ar dentro do carro. Gire o botão no sentido horário para aumentar o fluxo de ar ou no sentido anti-horário para diminuir o fluxo de ar. Velocidade do ventilador: O botão de velocidade do ventilador controla a velocidade do ventilador. Gire o botão no sentido horário para aumentar a velocidade do ventilador ou no sentido anti-horário para diminuir a velocidade do ventilador. Modo: O botão de modo permite que você selecione o modo desejado. Os modos disponíveis são: Auto: O carro ajustará automaticamente a temperatura e o fluxo de ar para manter um nível confortável. Cool (Frio): O carro soprará ar frio para dentro do carro. Heat: O carro soprará ar quente para dentro do carro. Defrost (Descongelamento): O carro soprará ar quente no para-brisa para descongelá-lo."}

DOCUMENT2 = {
    "Título": "Touchscreen",
    "Conteúdo": "O seu Googlecar tem uma grande tela sensível ao toque que fornece acesso a uma variedade de recursos, incluindo navegação, entretenimento e controle climático. Para usar a tela sensível ao toque, basta tocar no ícone desejado.  Por exemplo, você pode tocar no ícone \"Navigation\" (Navegação) para obter direções para o seu destino ou tocar no ícone \"Music\" (Música) para reproduzir suas músicas favoritas."}

DOCUMENT3 = {
    "Título": "Mudança de marchas",
    "Conteúdo": "Seu Googlecar tem uma transmissão automática. Para trocar as marchas, basta mover a alavanca de câmbio para a posição desejada.  Park (Estacionar): Essa posição é usada quando você está estacionado. As rodas são travadas e o carro não pode se mover. Marcha à ré: Essa posição é usada para dar ré. Neutro: Essa posição é usada quando você está parado em um semáforo ou no trânsito. O carro não está em marcha e não se moverá a menos que você pressione o pedal do acelerador. Drive (Dirigir): Essa posição é usada para dirigir para frente. Low: essa posição é usada para dirigir na neve ou em outras condições escorregadias."}

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]

# Pega a estrutura documents e coloca no formato DataFrame para poder acessar
# Irá organizar em duas colunas de título e conteúdo
df = pd.DataFrame(documents)
# Formato orginal
# print(df)
# Múda o título das colunas
df.columns = ["Title", "Content"]
# print(df)

def embed_fn(title, text):
    return genai.embed_content(
                model=model,
                content=text,
                title=title,
                task_type="RETRIEVAL_DOCUMENT"
        )["embedding"]
        #[embedding] = essa coluna será adicionada no DataFrame

# Cria nova coluna no DataFrame
# Apply do pandas: aplica a estrutura criada em cima que gera os embeddings, para cada linha do DataFrame
# lambda: roda linha por linha, como se fosse for
df["Embeddings"] = df.apply(lambda row: embed_fn(row["Title"], row["Content"]), axis=1)
print (df)

def generateAndSearchQuery(query, base, model):
    queryEmbed = genai.embed_content(
                model=model,
                content=query,
                task_type="RETRIEVAL_QUERY"
        )

    # Irá pegar a menor distancia entre os vetores dos embeddings da consulta e dos documentos empilhados, através de produto escalar
    produtos_escalares = np.dot(np.stack(df["Embeddings"]), queryEmbed["embedding"])
    # Vai pegar o id com maior similiaridade de contexto
    index = np.argmax(produtos_escalares)
    # Retorna o texto do Content do índice que foi achado
    return df.iloc[index]["Content"]

query = "Como faço para trocar a marcha em um carro da Google?"
queryResult = generateAndSearchQuery(query, df, model)
print("\n", queryResult , "\n")

# Utilizando embedding, que é uma forma fixa de resposta, com IA generativa, para adicionar criatividade
prompt = f"\nReescreva esse texto de uma forma mais descontraída, sem adicionar informações que não façam parte do texto: {queryResult}"
model_2 = genai.GenerativeModel("gemini-1.0-pro")
generation_config = {
    # Opções de resposta
    "candidate_count": 1,
    # Temperature: de 0 a 1. Quanto mais próximo de 1, mais criativo. Quanto mais próximo de 0, mais preciso.
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 35,
}
safety_settings = { 
    "HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    "HATE": "BLOCK_MEDIUM_AND_ABOVE",
    "SEXUAL": "BLOCK_MEDIUM_AND_ABOVE",
    "DANGEROUS": "BLOCK_MEDIUM_AND_ABOVE"
}
model = genai.GenerativeModel(
    model_name = 'gemini-pro',
    generation_config = generation_config,
    safety_settings = safety_settings
)

response = model_2.generate_content(prompt)
print(response.text)