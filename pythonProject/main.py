# 1. Import necessary modules
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_aws import BedrockLLM

# 2. Define the root directory where PDF files are located
root_directory = os.path.join(os.getcwd(), "pdf_files")

# 3. Wrap the index creation within a function
def hr_index(pdf_files):
    # Initialize an empty list to store loaded documents
    docs = []

    # Iterate through each PDF file
    for pdf_file in pdf_files:
        if pdf_file.lower().endswith(".pdf"):
            try:
                print(f"Processing {pdf_file}...")
                # Load data with PyPDFLoader
                data_load = PyPDFLoader(pdf_file)
                # Split the text based on character, tokens, etc.
                data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=300,
                                                            chunk_overlap=10)
                # Create embeddings
                data_embeddings = BedrockEmbeddings(credentials_profile_name='default',
                                                    model_id='amazon.titan-embed-text-v2:0')
                # Create Vector DB and store embeddings
                data_index = VectorstoreIndexCreator(text_splitter=data_split, embedding=data_embeddings,
                                                     vectorstore_cls=FAISS)
                # Create index for the document
                db_index = data_index.from_loaders([data_load])
                # Append the created index to the list
                docs.append(db_index)
                print(f"Processed {pdf_file}")
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
        else:
            print(f"File {pdf_file} is not a PDF. Skipping this file.")
            raise ValueError(f"File {pdf_file} is not a PDF. Cannot process this file.")

    return docs

# 4. Function to connect to Bedrock LLM
def hr_llm():
    llm = BedrockLLM(
        credentials_profile_name='default',
        model_id='anthropic.claude-v2',
        model_kwargs={
            "max_tokens_to_sample": 3000,
            "temperature": 0.1,
            "top_p": 0.5})
    return llm

# 5. Function to search user prompt and query the best match from Vector DB using LLM
def hr_rag_response(docs, question):
    rag_llm = hr_llm()
    responses = []
    for index in docs:
        try:
            response = index.query(question=question, llm=rag_llm)
            responses.append(response)
        except Exception as e:
            print(f"Error querying index: {e}")
    return responses

if __name__ == "__main__":
    # Get the list of PDF files to process
    pdf_files_to_process = [os.path.join(root_directory, file) for file in os.listdir(root_directory) if file.lower().endswith(".pdf")]

    print(f"Found {len(pdf_files_to_process)} PDF files to process")

    # Process PDF files to create indexes
    indexed_docs = hr_index(pdf_files_to_process)
    print(f"Total documents indexed: {len(indexed_docs)}")

    # Example of querying with a question
    """
    question = "What are the leave policies?"
    responses = hr_rag_response(indexed_docs, question)
    print(f"Responses to '{question}': {responses}")
    """
