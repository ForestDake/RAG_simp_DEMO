from langchain_community.document_loaders import (PyPDFLoader)
from langchain.text_splitter import (RecursiveCharacterTextSplitter)
from langchain_community.vectorstores import Chroma
import os
from langchain_community.embeddings import HuggingFaceEmbeddings

path_docfolder = "/Users/qicao/Documents/GitHub/RAG_simp_DEMO/data/AutomobileIndustry_raw"
path_db = "/Users/qicao/Documents/GitHub/RAG_simp_DEMO/data/DB"

#Choose the embedding model
#model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = "sentence-transformers/sentence-t5-large"
embedding = HuggingFaceEmbeddings(model_name=model_name)


def read_pdf_files_in_folder_onebyone_and_Store(path_docfolder, path_db, embedding):
    # Iterate over all files in the folder
    for filename in os.listdir(path_docfolder):
        print(filename)
        if filename.endswith('.pdf'):  # Check if the file is a PDF
            file_path = os.path.join(path_docfolder, filename)
            print(f"Reading file: {file_path}")

            # Open the PDF file
            with open(file_path, 'rb') as file:
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=260,
                    chunk_overlap=20,
                )
                docs = text_splitter.split_documents(pages)
                # Facility Step 3:用特定模型做embedding
                db2 = Chroma.from_documents(docs, embedding, persist_directory=path_db)
                print("Successfully save the embedding into DB")
    return True

read_pdf_files_in_folder_onebyone_and_Store(path_docfolder, path_db, embedding)

#------------------Now from here we need a chat to discuss with me --------------------
