# coding: utf-8
import nest_asyncio
nest_asyncio.apply()
import openai

openai.api_key = ""
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# 環境変数を設定
import os
os.environ["OPENAI_API_KEY"] = openai.api_key

def faiss_query(query_base):
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings)
    query = query_base
    embedding_vector = embeddings.embed_query(query)
    docs_and_scores = db.similarity_search_by_vector(embedding_vector)
    res = db.similarity_search_with_score(query, k=30)
    socore=res[0][1]
    return docs_and_scores,socore







