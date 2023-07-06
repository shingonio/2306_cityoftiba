import numpy as np
from langchain.document_loaders import TextLoader
import torch
from langchain.document_loaders import TextLoader
from transformers import BertJapaneseTokenizer, BertModel
from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    ResponseSynthesizer
)
from llama_index.indices.document_summary import GPTDocumentSummaryIndex
from langchain.chat_models import ChatOpenAI
# Used to create the dense document vectors.
import openai
openai.api_key = "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"
# 環境変数を設定
import os
os.environ["OPENAI_API_KEY"] = openai.api_key




import faiss
import numpy as np
class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()
        device = "cpu"
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


# Used to do vector searches and display the results.

def vector_search(query, model, index, num_results=100):
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector.detach().numpy()).astype("float32"), k=num_results)
    return D, I


def id2details(docs, I):
    res=[]
    for idx in I[0]:
       res.append(docs[idx-1])
    return res



def summarize_Text(data):

        prompt = f"以下のテキストを要約してください: {data}"
        response = openai.Completion().create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.0,
            max_tokens=600,
            top_p=1.0,
            frequency_penalty=1.0,
            presence_penalty=0.0,
        )

        return response['choices'][0]['text']
def completion(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=text,
        max_tokens=600,
        temperature=0.7,
        frequency_penalty=1.0,
    )
    return response['choices'][0]['text']
def gen_summary(text):

    summary=summarize_Text(text)

    return summary

def main():
    import os
    import scipy
    root_dir = './data3'
    docs = []
    docid_list=[]
    model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens")
    docid=1

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
               # out_suumary=gen_summary(loader.load_and_split()[0].page_content)
                #docs.append(out_suumary)
                docs.append(loader.load_and_split()[0].page_content)
                docid_list.append(docid)
                docid=docid+1
            except Exception as e:
                pass

    sentence_vectors=model.encode(docs)
    index = faiss.IndexFlat(sentence_vectors.shape[1])
    index = faiss.IndexIDMap(index)
    index.add_with_ids(sentence_vectors.detach().numpy(), docid_list)
    faiss.write_index(index, "faiss_sbert.index")

if __name__ == '__main__':
    main()
