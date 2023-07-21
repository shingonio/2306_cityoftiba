
import csv
import os
import time
from typing import List, Tuple, Callable, Any, Dict, Optional
import logging
import sys

import faiss
import numpy as np
import psutil
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import torch

class ScalableSemanticSearch:
    def __init__(self, device="cpu"):
        self.device = device
        self.model0 = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2", device=self.device
        )
        self.model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

        self.dimension = self.model.get_sentence_embedding_dimension()
        self.quantizer = None
        self.index = None
        self.hashmap_index_sentence = None

        log_directory = "log"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        log_file_path = os.path.join(log_directory, "scalable_semantic_search.log")

        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
        )
        logging.info("ScalableSemanticSearch initialized with device: %s", self.device)

    @staticmethod
    def calculate_clusters(n_data_points: int) -> int:
        return max(2, min(n_data_points, int(np.sqrt(n_data_points))))

    def encode(self, data: List[str]) -> np.ndarray:
        embeddings = self.model.encode(data)
        self.hashmap_index_sentence = self.index_to_sentence_map(data)
        return embeddings.astype("float32")

    def build_index(self, embeddings: np.ndarray) -> None:
        n_data_points = len(embeddings)
        if (
            n_data_points >= 1500
        ):
            # Adjust this value based on the minimum number of data points required for IndexIVFPQ
            self.quantizer = faiss.IndexFlatL2(self.dimension)
            n_clusters = self.calculate_clusters(n_data_points)
            self.index = faiss.IndexIVFPQ(
                self.quantizer, self.dimension, n_clusters, 8, 4
            )
            logging.info("IndexIVFPQ created with %d clusters", n_clusters)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            logging.info("IndexFlatL2 created")

        if isinstance(self.index, faiss.IndexIVFPQ):
            self.index.train(embeddings)
        self.index.add(embeddings)
        logging.info("Index built on device: %s", self.device)

    @staticmethod
    def index_to_sentence_map(data: List[str]) -> Dict[int, str]:
        return {index: sentence for index, sentence in enumerate(data)}

    @staticmethod
    def get_top_sentences(
        index_map: Dict[int, str], top_indices: np.ndarray
    ) -> List[str]:
        return [index_map[i] for i in top_indices]

    def search(self, input_sentence: str, top: int) -> Tuple[np.ndarray, np.ndarray]:
        vectorized_input = self.model.encode(
            [input_sentence], device=self.device
        ).astype("float32")
        D, I = self.index.search(vectorized_input, top)
        return I[0], 1 - D[0]

    def save_index(self, file_path: str) -> None:
        if hasattr(self, "index"):
            faiss.write_index(self.index, file_path)
        else:
            raise AttributeError(
                "The index has not been built yet. Build the index using `build_index` method first."
            )

    def load_index(self, file_path: str) -> None:
        if os.path.exists(file_path):
            self.index = faiss.read_index(file_path)
        else:
            raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")

    @staticmethod
    def measure_time(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return elapsed_time, result

    @staticmethod
    def measure_memory_usage() -> float:
        process = psutil.Process(os.getpid())
        ram = process.memory_info().rss
        return ram / (1024**2)

    def timed_infer(self, query: str, top: int) -> Tuple[float, float]:
        start_time = time.time()
        _, _ = self.search(query, top)
        end_time = time.time()
        elapsed_time = end_time - start_time
        memory_usage = self.measure_memory_usage()
        logging.info(
            "Inference time: %.2f seconds on device: %s", elapsed_time, self.device
        )
        logging.info("Inference memory usage: %.2f MB", memory_usage)
        return elapsed_time, memory_usage

    def timed_load_index(self, file_path: str) -> float:
        start_time = time.time()
        self.load_index(file_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(
            "Index loading time: %.2f seconds on device: %s", elapsed_time, self.device
        )
        return elapsed_time


def pickup_context(load_path):

    soup = BeautifulSoup(open(load_path, encoding="utf8"), "html.parser")
    title = soup.find('title').text
    body = soup.get_text(strip="true")
    for meta_tag in soup.find_all('meta', attrs={'name': 'keywords'}):
        meta_keyword=meta_tag.get('content')
    for meta_tag in soup.find_all('meta', attrs={'name': 'description'}):
        meta_description=meta_tag.get('content')
    if(len(meta_keyword)==0):
        body=""
    return meta_keyword,meta_description,title,body

def main():
    root_dir = './chiba-city.mamafre.jp'
    title_list = []
    meta_keyword_list = []
    meta_description_list = []
    body_list = []
    docid_list = []
    docid = 1
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            ext_without_dot = os.path.splitext(name)[1][1:]
            if (ext_without_dot == "html"):
                try:
                    load_path = os.path.join(root, name)
                    meta_keyword, meta_description, title, body = pickup_context(load_path)
                    title_list.append(title)
                    body_list.append(body)
                    meta_keyword_list.append(meta_keyword)
                    meta_description_list.append(meta_description)
                    docid_list.append(str(docid))
                    docid = docid + 1
                except Exception as e:
                    pass

    semantic_search = ScalableSemanticSearch(device="cpu")
    embeddings = semantic_search.encode(title_list+body_list)
    semantic_search.build_index(embeddings)

    # ユーザからの入力を受け取り、変数commandに代入ge-muwo
    command = input("You: ")

    while True:
        query = command
        top = 5
        top_indices, top_scores = semantic_search.search(query, top)
        top_sentences = ScalableSemanticSearch.get_top_sentences(semantic_search.hashmap_index_sentence, top_indices)
        print(top_sentences[0][:300])
        print(meta_description_list[top_indices[0]])
        print(top_sentences[1][:300])
        print(meta_description_list[top_indices[1]])
        print(top_sentences[2][:300])
        print(meta_description_list[top_indices[2]])

        command= input("You: ")

        if command == "exit":
            break

if __name__ == '__main__':
    main()
