import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate  # Importa PromptTemplate
from deep_translator import GoogleTranslator


pdf_path = "2020-Scrum-Guide-US.pdf" # Reemplaza con la ruta a tu PDF
model_path = "Lexi-Llama-3-8B-Uncensored_Q4_K_M.gguf" # Ajusta según la ubicación de tu modelo
persist_directory = "chroma_db"

def crear_base_de_conocimiento():
    # 1. Cargar el documento PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Fragmentar el texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 3. Crear incrustaciones con la nueva clase
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # 4. Crear y persistir la base de datos vectorial Chroma
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    print("Base de conocimiento creada y persistida en chroma_db")

def consultar_base_de_conocimiento(query):

    try:
        translated_query = GoogleTranslator(source='es', target='en').translate(query)
        print(f"Pregunta traducida (a inglés): {translated_query}")
    except Exception as e:
        print(f"Error al traducir la pregunta: {e}")
        translated_query = query

    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_batch=512,
        n_ctx=8192,
        callback_manager=None,
        verbose=False,
    )

    # 1. Crea una plantilla de prompt para guiar la respuesta del LLM
    template = """Use the following pieces of context to answer the user's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answer only the user's question directly, without adding extra comments, questions, or conversation.

    {context}

    Question: {question}
    Helpful Answer:"""

    qa_prompt = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    # 2. Configura RetrievalQA con la plantilla de prompt
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": qa_prompt}  # Pasa la plantilla a la cadena
    )

    # 3. Ejecuta la consulta
    result_en = qa.invoke({"query": translated_query})

    if isinstance(result_en, dict) and 'result' in result_en:
        result_text_en = result_en['result']
    else:
        result_text_en = str(result_en)

    try:
        result_es = GoogleTranslator(source='en', target='es').translate(result_text_en)
    except Exception as e:
        print(f"Error al traducir la respuesta: {e}")
        result_es = result_text_en

    print(f"Pregunta original: {query}")
    print(f"Respuesta (en español): {result_es}")

if __name__ == "__main__":
    if not os.path.exists(persist_directory):
        crear_base_de_conocimiento()
    else:
        print("La base de conocimiento ya existe. Omitiendo la creación.")

    while True:
        pregunta = input("Ingresa tu pregunta (o 'salir' para terminar): ")
        if pregunta.lower() == "salir":
            break
        consultar_base_de_conocimiento(pregunta)
