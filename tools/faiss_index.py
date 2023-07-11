# coding: utf-8
import nest_asyncio
nest_asyncio.apply()
import openai

openai.api_key = "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"
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

# 環境変数を設定
import os
os.environ["OPENAI_API_KEY"] = openai.api_key
from langchain.text_splitter import CharacterTextSplitter

root_dir = './data3'
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
db.save_local("faiss_index")






