from langchain_community.document_loaders import (PyPDFLoader)
#from langchain.text_splitter import (RecursiveCharacterTextSplitter)
import os
import numpy as np
from ltp import StnSplit
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)

path_docfolder = "/Users/qicao/Documents/GitHub/RAG_langchain/data/AutomobileIndustry_raw"
path_db = "/Users/qicao/Documents/GitHub/RAG_simp_DEMO/data/DB"

#Choose the embedding model
#model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = "sentence-transformers/sentence-t5-large"

embedding_function = SentenceTransformerEmbeddings(model_name=model_name)

#here is the setting for the size of chunk, 100 is one article only one chunk
THRESHOLD = 70

class SemanticParagraphSplitter:
    def __init__(self, threshold=THRESHOLD, model_path=model_name):
        self.threshold = threshold
        self.model = SentenceTransformer(model_path)

    @staticmethod
    def cut_sentences(text):
        sentences = StnSplit().split(text)
        return sentences

    @staticmethod
    def combine_sentences(sentences, buffer_size=2):
        # Go through each sentence dict
        for i in range(len(sentences)):

            # Create a string that will hold the sentences which are joined
            combined_sentence = ''

            # Add sentences before the current one, based on the buffer size.
            for j in range(i - buffer_size, i):
                # Check if the index j is not negative (to avoid index out of range like on the first one)
                if j >= 0:
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += sentences[j]['sentence'] + ' '

            # Add the current sentence
            combined_sentence += sentences[i]['sentence']

            # Add sentences after the current one, based on the buffer size
            for j in range(i + 1, i + 1 + buffer_size):
                # Check if the index j is within the range of the sentences list
                if j < len(sentences):
                    # Add the sentence at index j to the combined_sentence string
                    combined_sentence += ' ' + sentences[j]['sentence']

            # Then add the whole thing to your dict
            # Store the combined sentence in the current sentence dict
            sentences[i]['combined_sentence'] = combined_sentence

        return sentences

    def build_sentences_dict(self, sentences):
        indexed_sentences = [{'sentence': x, 'index': i} for i, x in enumerate(sentences)]
        combined_sentences = self.combine_sentences(indexed_sentences)

        embeddings = self.model.encode([x['combined_sentence'] for x in combined_sentences], normalize_embeddings=True)

        for i, sentence in enumerate(combined_sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]

        return combined_sentences

    @staticmethod
    def calculate_cosine_distances(sentences):
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['combined_sentence_embedding']
            embedding_next = sentences[i + 1]['combined_sentence_embedding']

            # Calculate cosine similarity
            # similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
            similarity = embedding_current @ embedding_next.T
            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the list
            distances.append(distance)

            # Store distance in the dictionary
            sentences[i]['distance_to_next'] = distance

        # Optionally handle the last sentence
        # sentences[-1]['distance_to_next'] = None  # or a default value

        return distances, sentences

    def calculate_indices_above_thresh(self, distances):
        breakpoint_distance_threshold = np.percentile(distances, self.threshold)
        # The indices of those breakpoints on your list
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
        return indices_above_thresh

    @staticmethod
    def cut_chunks(indices_above_thresh, sentences):
        # Initialize the start index
        start_index = 0

        # Create a list to hold the grouped sentences
        chunks = []

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks

    def split(self, text):
        single_sentences = (self.cut_sentences(text)) #Pre-split with standard function
        print(f"{len(single_sentences)} single sentences were found")
        chunks = self.split_passages(single_sentences)
        return chunks

    def split_passages(self, passages):
        combined_sentences = self.build_sentences_dict(passages)
        distances, sentences = self.calculate_cosine_distances(combined_sentences)

        indices_above_thresh = self.calculate_indices_above_thresh(distances)
        chunks = self.cut_chunks(indices_above_thresh, sentences)
        return chunks

def read_pdf_files_in_folder_onebyone_and_Store(path_docfolder, path_db, embedding):
    # Iterate over all files in the folder
    for filename in os.listdir(path_docfolder):
        #print(filename)
        if filename.endswith('.pdf'):  # Check if the file is a PDF
            file_path = os.path.join(path_docfolder, filename)
            print(f"Reading file: {file_path}")

            # Open the PDF file
            with open(file_path, 'rb') as file:
                loader = PyPDFLoader(file_path)
                pages_pypdf = loader.load()
                pages = pages_pypdf[0].page_content

                text_splitter = SemanticParagraphSplitter(threshold=THRESHOLD)
                # text_splitter = RecursiveCharacterTextSplitter(
                #     chunk_size=260,
                #     chunk_overlap=20,
                # )
                docs = text_splitter.split(pages)

                # Facility Step 3:用特定模型做embedding
                #db2 = Chroma.from_documents(docs, embedding, persist_directory=path_db)
                db2 = Chroma.from_texts(docs, embedding, persist_directory=path_db)
                print("Successfully save the embedding into DB")
    return True

read_pdf_files_in_folder_onebyone_and_Store(path_docfolder, path_db, embedding_function)

#------------------Now from here we need a chat to discuss with me --------------------
