import os

from llmsherpa.readers import LayoutPDFReader
import getpass
from IPython.core.display import display, HTML

from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch

from langchain.embeddings import HuggingFaceEmbeddings

from google.colab import drive
drive.mount('/content/drive')

MONGODB_ATLAS_CLUSTER_URI = ""

# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = "langchain_db"
COLLECTION_NAME = "upsc_chunking"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "index_name"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# specify embedding model (using huggingface sentence transformer)
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

import re

def extract_metadata(pdf_url):
  match = re.search(r"/content/drive/MyDrive/([^/]+)/([^/]+)/([^/]+)/(\d+)/(.*?)/(.*?)\.pdf", pdf_url)
  if match:
    return {
      "board": match.group(1),
      "book": match.group(2),
      "language": match.group(3),
      "class": int(match.group(4)),
      "subject": match.group(5),  # Handle "And" case
      "chapter": match.group(6).replace(".pdf", ""),
    }
  else:
    print("Failed to extract metadata from the provided URL format.")
    return None

llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
# pdf_url = "/content/drive/MyDrive/Advision_file/Ncert Chapters/12_1.pdf"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)

class Document:
  def __init__(self, page_content, metadata):
    self.page_content = page_content
    self.metadata = metadata

docs = []

# Traverse through the extracted folders
for root, dirs, files in os.walk("/content/drive/MyDrive/cbse"):
    # print("Folder:", root)
    # Print the names of files in each folder
    for file_name in files:
        pdf_url =  os.path.join(root, file_name)
        print(pdf_url)
        # print("File:", os.path.join(root, file_name))
        doc = pdf_reader.read_pdf(pdf_url)
        metadata = extract_metadata(pdf_url)

        for chunk in doc.chunks():
            chunk_text = chunk.to_text()
            docs.append(Document(page_content=chunk_text, metadata=metadata))

vector_search = MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=embeddings,
            collection=MONGODB_COLLECTION,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        )