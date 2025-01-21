from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama.chat_models import ChatOllama
from langchain_chroma import Chroma
#from langchain_openai import ChatOpenAI
from base64 import b64decode

gpt4all_embd = GPT4AllEmbeddings()
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    persist_directory="./multimodal_vector_db", 
    embedding_function=gpt4all_embd)

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text


    # construct prompt with context (including images)
    system_prompt = """You are an assistant tasked with answering the question 
    base only on the following context, which can include text, tables and images"""

    prompt_template = f"""
    Here is the provided context and question to answer the question:
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        
        prompt_content.append(
            {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + docs_by_type["images"][0]}
            }
        )
    """
    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )
    """

    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content= system_prompt),
            HumanMessage(content=prompt_content),
        ]
    )


chain = (
    {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | ChatOllama(model= "llama3.2-vision:11b")
    | StrOutputParser()
)

