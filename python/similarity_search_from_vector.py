from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from flashrank import Ranker, RerankRequest
import csv

# from google.colab import drive
# drive.mount('/content/drive')

MONGODB_ATLAS_CLUSTER_URI = "mongodb+srv://patelniraj313:JZW57Er7zqdKKCBR@cluster0.agb4hnr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = "langchain_db"
COLLECTION_NAME = "test"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "index_name"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# insert the documents in MongoDB Atlas with their embedding
vector_search = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

query = "What is the responsibilty of democratic government?"
print(query)

def query_data(query):
  docs = vector_search.similarity_search(query, K=10)
  return docs

# data from retriver
def formatting_data(docs):
  formatted_data = [{"text": doc.page_content} for doc in docs]
  return formatted_data

# load the reranking model
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")

def reranking(query, formatted_data):
  rerankrequest = RerankRequest(query=query, passages=formatted_data)
  results = ranker.rerank(rerankrequest)
  return results

import collections
text_list = collections.OrderedDict()

def formatted_answer(results):
  answer = [item["text"] for item in results]
  for i in range(len(answer)):
    answer[i] = answer[i].replace("\n", " ")
  return answer

query = []
query_contents = []

with open('/content/100_questions_split_question.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for row in reader:
    question = row[0]
    print(question)
    docs = query_data(question)
    formatted_data = formatting_data(docs)

    reranked_results = reranking(question, formatted_data)

    answer = formatted_answer(reranked_results)

    query.append(question)
    query_contents.append(answer[0])

# Create the CSV file
with open("100_question_with_reranked_ans.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["query", "query_content"])  # Header row

    # Write data rows
    for query, query_content in zip(query, query_contents):
        writer.writerow([query, query_content])

print("CSV file created successfully!")