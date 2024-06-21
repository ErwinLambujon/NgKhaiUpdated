import os
from langchain_community.document_loaders import PyPDFLoader
from concurrent.futures import ThreadPoolExecutor

# Specify the root directory where you want to search for PDF files
root_directory = os.path.join(os.getcwd(), "pdf_files")

# Set the batch size (number of files to process in each batch)
batch_size = 100  # Adjust batch size as per your needs

# Initialize an empty list to store loaded documents
docs = []

# Function to process a single PDF file
"""
    We created this function for a single document
    because this a simple task and we can reuse
    this function in the more complicated task
    which is the appending of multiple documents.   
"""
def process_single_pdf(pdf_file_path):
    try:
        pdf_loader = PyPDFLoader(pdf_file_path)
        loaded_docs = pdf_loader.load()
        print(f"Loaded {len(loaded_docs)} documents from {pdf_file_path}")
        return loaded_docs
    except Exception as e:
        print(f"Error loading PDF {pdf_file_path}: {str(e)}")
        return []

"""
    Function to process a batch of PDF files
    Here we use extend function instead of append function for formality purposes
    since in latter part we use extend function especially in getting the list
    of files to be processed since if we use append function there this will cause
    an error in processing the file because append function will add it as a nested
    list each time you add a file path to the list.
"""
def process_pdf_batch(pdf_files):
        batch_docs = []
        for pdf_file_path in pdf_files:
            if pdf_file_path.lower().endswith(".pdf"):
                try:
                    pdf_loader = PyPDFLoader(pdf_file_path)
                    loaded_docs = pdf_loader.load()
                    batch_docs.extend(loaded_docs)
                    print(f"Loaded {len(loaded_docs)} documents from {pdf_file_path}")
                except Exception as e:
                    print(f"Error loading PDF from file {pdf_file_path}: {str(e)}")
            else:
                # Explicitly handle non-PDF files
                print(f"File {pdf_file_path} is not a PDF. Skipping this file.")
        return batch_docs

if __name__ == "__main__":
    # Get the list of PDF files to process
    pdf_files_to_process = []
    for root, dirs, files in os.walk(root_directory):
        pdf_files_to_process.extend([os.path.join(root, file) for file in files if file.lower().endswith(".pdf")])

    print(f"Found {len(pdf_files_to_process)} PDF files to process")

    """ 
        Create a ThreadPoolExecutor for parallel processing
        This is an efficient approach especially with large
        number of files since this processes can multiple PDF files
    """
    with ThreadPoolExecutor() as executor:
        total_files = len(pdf_files_to_process)
        processed_files = 0

        # Iterate through the PDF files in batches
        for i in range(0, total_files, batch_size):
            batch = pdf_files_to_process[i:i+batch_size]
            batch_docs = list(executor.map(process_pdf_batch, [batch]))
            for batch_result in batch_docs:
                if batch_result is not None:
                    docs.extend(batch_result)
                    processed_files += len(batch_result)
                    print(f"Processed {processed_files} / {total_files} files")

    # Printing the number of documents loaded for debugging purposes.
    print(f"Total documents loaded: {len(docs)}")