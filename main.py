from langchain_chroma.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate #type: ignore
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv() 

CAMINHO_DB = "db"

prompt_template = """
responda a pergunta do usuario:
{pergunta}

com base nessas informacoes abaixo:

{base_conhecimento}"""

def perguntar():
    pergunta = input("Digite sua pergunta: ")

    funcao_embedding = OpenAIEmbeddings()
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

    resultados = db.similarity_search_with_relevance_scores(pergunta,k=3)
    if len(resultados) == 0 or resultados[0][1] < 0.7:
        print("Desculpe, não sei a resposta para essa pergunta.")
        return
    texto_resultado = []
    for resultado in resultados:
        texto = resultado[0].page_content
        texto_resultado.append(texto)

    base_conhecimento = "\n".join(texto_resultado) 

    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"pergunta": pergunta, "base_conhecimento": base_conhecimento})

    modelo = ChatOpenAI()
    texto_resposta = modelo.invoke(prompt).content
    print("resposta da IA", texto_resposta)

perguntar()

