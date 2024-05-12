<div align="center">
# Google AI Studio - Gemini API
A brief introduction to Google AI Studio

[Tecnologias](#tecnologias)
[Projetos](#projetos)
</div>

## Tecnologias: 
* [Python](https://www.python.org/doc/)
* [Gemini API](https://ai.google.dev/)

## Projetos:
* Python Review
  * Um jogo que consome uma API que traz diversas linguagens de programação, frameworks. Será sorteado um framework, e será oferecida apenas a dica de qual é. O usuário deve tentar acerta-la.
* [Chat-bot](#chat-bot)
  * Integração com a AI generativa do Google, Gemini. Utilização da API no código, podendo ter chats pelo terminal.
* [Projeto: Film-indicator](#film-indicator)
  * Um indicador de filmes através de perguntas pré-estabelecidas na IA generativa Gemini.
* [Embeddings](#embeddings)
  * Criação de embeddings para receber um documento, receber uma consulta e buscar essa consulta nesse documento através de embeddings.

Para os projetos de chat-bot, embeddings, film-indicator, é necessário a instalação do SDK da Google generativa
~~~ bash
pip install -q -U google-generativeai
~~~

Deve-se cadastrar também a Google API Key para rodar o projeto. A KEY pode ser adquirida nesse [link](https://aistudio.google.com/app/prompts/new_chat).
A KEY está inserida nos códigos com as seguintes linhas de comando:
~~~python
    import os
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
~~~
Após isso, criar na pasta um arquivo .env como escrito abaixo:

~~~ 
    GOOGLE_API_KEY=[KEY_RECEBIDA_DO_GOOGLE_AI_STUDIO]
~~~

### Chat-bot:
É configurada a API do Google AI:
~~~python
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    generation_config = {
        "candidate_count": 1,
        "temperature": 0.8,
    }
~~~
No generation_config é possível ajustar os parâmetros Temperature, Top K, Top P


Para listar os modelos suportados:
~~~python
    Listar os modelos generativos disponíveis: 
    for modelSupported in genai.list_models():
        if 'generateContent' in modelSupported.supported_generation_methods:
            print(modelSupported.name)
~~~

Para ajustar os níveis de segurança:
* Os níveis possíveis de bloqueio são: BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE
~~~python
    safety_settings = { 
        "HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HATE": "BLOCK_MEDIUM_AND_ABOVE",
        "SEXUAL": "BLOCK_MEDIUM_AND_ABOVE",
        "DANGEROUS": "BLOCK_MEDIUM_AND_ABOVE"
    }
~~~

Após isso é inicializado a API: 
~~~python
        model = genai.GenerativeModel(
        model_name = 'gemini-pro',
        generation_config = generation_config,
        safety_settings = safety_settings
    )
    chat = model.start_chat(history=[])
~~~

O chat é inicializado e entra em looping até a condição de parada:
~~~python
    print("Insira o prompt. \nQuando finalizar as consultas, digite Fim\n")
    prompt = input("Esperando prompt: ")

    while prompt != "Fim":
        response = chat.send_message(prompt)
        print("Resposta: ", response.text)
        print("--------------------------------\n")
        prompt = input("Esperando prompt: ")
~~~

### Film-indicator
O código de indicação de filmes é quase idêntico ao chat-bot.
Porém são dadas intruções para o chat-bot em como se comportar, e como apresentar as quatro perguntas que servirá de parâmetro para indicar o filme
~~~ python
    response = chat.send_message("(IDEIA: você analisará quatro respostas minhas, para quatro perguntas que você vai fazer para mim, de forma criativa. Somente no final das quatro perguntas você me indicará um filme ou uma série, com base nessas quatro respostas. REQUISITOS: NÃO FAÇA DUAS PERGUNTAS DE UMA VEZ, faça uma pergunta e espere a resposta e NÃO indique o filme antes das quatro perguntas. PERGUNTAS: PRIMEIRA: qual gênero você gostaria de assistir? SEGUNDA: cite três filmes desse mesmo gênero que você gostou de ver? TERCEIRA: qual tema você gostaria de assistir? QUARTA: você prefere uma série ou filme?) Faça a primeira pergunta agora")

    for i in range(4):
        print("\n", response.text)
        prompt = input("\nEsperando prompt: ")
        response = chat.send_message(prompt)
        
    print("\n", response.text)
~~~

### Embeddings
