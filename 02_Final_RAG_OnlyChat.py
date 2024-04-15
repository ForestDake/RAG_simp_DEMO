# Example: reuse your existing OpenAI setup
from openai import OpenAI
import os
from langchain.chat_models import ChatOpenAI
#from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
##from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

path_db = "/Users/qicao/Documents/GitHub/RAG_simp_DEMO/data/DB"

#Define a function to combine Query and the data found from RAG Vector DB.
def augment_prompt(query:str):
    results = vectorstore.similarity_search(query,k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    augment_prompt = f"""Using the contexts below, answer the query:
    contexts:
    {source_knowledge}
    query:{query}"""
    return augment_prompt
    #Output is the original query with top 3 result from database

def llm(query, history=[], user_stop_words=[]):  # 调用api_server
    try:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=augment_prompt(query)),
        ]
        res = chat(messages)
        #print("-----Answer of the xPAI to your question is------")
        #print(res.content)
        content = res.content
        return content
    except Exception as e:
        return str(e)
    #The output is the feedback from LLM

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
os.environ["OPENAI_API_KEY"] = "not-needed"
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"

chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"]
)

from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)

#Facility Step 3:用特定模型做embedding
from langchain.embeddings import HuggingFaceEmbeddings
model_name = "sentence-transformers/sentence-t5-large"
#model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=model_name)

# load DB from disk
vectorstore = Chroma(persist_directory=path_db, embedding_function=embedding)

#------------------Now from here we need a chat to discuss with me --------------------

def agent_execute(query, chat_history=[]):
    global augment_prompt, llm

    agent_scratchpad = ''  # agent执行过程
    while True:
        # 1）触发llm思考下一步action
        prompt = augment_prompt(query)
        #print("Promt is")
        #print(prompt)
        response = llm(prompt, user_stop_words=['Observation:'])
        print("-----The Answer of xPAI is ---------")
        print(response)
        chat_history.append((query, response))
        return True, response, chat_history
    return False, "", ""

def agent_execute_with_retry(query, chat_history=[], retry_times=3):
    for i in range(retry_times):
        success = False
        success, response, chat_history = agent_execute(query, chat_history=chat_history)
        #print(success)
        return success, response, chat_history

#Chat Step 1:
my_history = []
while True:
    query = input('query:')
    success, response, my_history = agent_execute_with_retry(query, chat_history=my_history)
    my_history = my_history[-10:]





