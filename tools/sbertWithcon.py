from langchain.document_loaders import TextLoader
import torch
from langchain.document_loaders import TextLoader
from transformers import BertJapaneseTokenizer, BertModel
import os
import scipy


class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest",
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)







def query(query_str,sentece_vectors):
    model= SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens")
    query =query_str
    queries = [query]
    query_embeddings = model.encode(queries)

    # Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
    number_top_matches = 3  # @param {type: "number"}
    res_list=[]
    score_list=[]
    for query, query_embedding in zip(queries, query_embeddings):
        query_embeddings_clone=query_embedding.detach().numpy()
        sentence_vectors_clone=sentence_vectors.detach().numpy()
        distances = scipy.spatial.distance.cdist([query_embeddings_clone], sentence_vectors_clone, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        for idx, distance in results[0:number_top_matches]:
            res_list.append(docs[idx].strip())
            score_list.append(1-distance)
    return res_list,score_list





