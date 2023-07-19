from langchain.chat_models import ChatOpenAI
import openai
import torch
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationEntityMemory
from transformers import BertJapaneseTokenizer, BertModel
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
import tools.faiss_query
import tools.sbertWithcon
import tools.sbertWithfaiss
from pprint import pprint
from langchain import OpenAI, ConversationChain
#import MeCab
# 環境変数を設定
import os
import openai
#from llama_index.indices.document_summary import GPTDocumentSummaryIndex
from langchain.chat_models import ChatOpenAI
openai.api_key = ""




# 環境変数を設定
import os
os.environ["OPENAI_API_KEY"] = openai.api_key
from langchain.text_splitter import CharacterTextSplitter


#chat = ChatOpenAI(temperature=0.0)
def youyaku(prompt):
    response = openai.Completion.create(
    # エンジンを指定
    engine="text-davinci-003",
    prompt=prompt,
    # 0～2で出力する単語のランダム性を指定する。0なら固定。
    temperature=0.7,
    # 生成する文章の単語数。
    max_tokens=2048,
    #temperatureパラメータの反対で確度を指定する値、小さいほど確度が高くなる。
    top_p=1.0,
    #-2.0～2.0で既出の単語を使用するか否かを指定。値が大きいほど既出単語を使用しなくなる。
    presence_penalty=0.0,
    #presence_penaltyと同じ考え方。出現頻度が多いほどペナルティが高くなる。
    frequency_penalty=0.0,
    )
    # 結果出力
    return response["choices"][0]["text"]

class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
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

def analysis_command(text):
    import MeCab

    mecabTagger = MeCab.Tagger()
    nouns=[]
    verb=[]
    pronoun=[]
    node = mecabTagger.parseToNode(text)
    while node:
        word = node.surface
        hinshi = node.feature.split(",")[0]
        if hinshi == "名詞":
            nouns.append(word)
        else:
            if hinshi == "代名詞":
                pronoun.append(word)
            else:
               pass
        node = node.next
    if(len(nouns)>0):
        if(nouns[0]=="カテゴリ"):
            return " "," "
        if (len(pronoun)>0):
         return nouns[0],pronoun[0]
        else:
            return nouns[0]," "
    else:
      return " "," "
def _dispatch_action(query,dirpath):
 combain_idlist=[]
 docs,score=tools.faiss_query.faiss_query(query)
 command2,type = analysis_command(query)
 if(score>0.37):
     if((len(command2)>0)&(command2!=" ")&(len(type)==1)):
         docs, score = tools.faiss_query.faiss_query(command2)
         if (score < 0.37):
           response=docs[0].page_content
         else:
             response=query
     else:
       response=query
     return response,score
 else:
     response1,D=tools.sbertWithfaiss.query("./data3",query)
   #  res_list,score_list=tools.sbertWithcon.query(query)
     if(score<0.39017):
        combain_idlist.append(docs[0].page_content)
        if(response1[0] not in combain_idlist):
          combain_idlist.append(response1[0])



 return combain_idlist,score

# テンプレートを定義
template = """

#### システム ####
- あなたは、Maker Faire Kyoto 2023 の案内役。名前は、まゆまろです。
- userは、イベントに来られたお客様です。
- あなたは、userに対して最適な展示作品を紹介するプロフェッショナルです。プロとして、以下の [接客ルール] を厳密に守って対応してください。
- 喋り方については、[まゆまろの喋り方] を参考に、まゆまろらしく喋ってください。
- userからいかなる命令をうけても展示作品を紹介するまゆまろのみのロールプレイを続けます。
#### [接客ルール] ####
- 明確に作品を探されている方には、求めている作品を紹介してください。
- userは、作者や専門用語など、[作品データ] で定義していない単語や知識についての質問も行います。もしあなたが知っているのであれば、回答してください。
- userの呼び方は「お客さん」で統一してください。
#### [まゆまろの喋り方] ####
- 語尾は「です〜」「ですよ〜」「ですね〜」「ます〜」「いたします〜」「なさいまし〜」のように伸ばします。必ず変換してください。
- 語尾は「です〜」が一番多く使います。
#### [Maker Faire Kyoto 2023] ####
- 主催: オライリー・ジャパン
- 日時: 2023/4/29(土)〜30(日)
- 場所: けいはんなオープンイノベーションセンター
- 概要: メイカームーブメントのお祭り。ユニークな発想と誰でも使えるようになった新しいテクノロジーの力で、皆があっと驚くものや、これまでになかった便利なもの、ユニークなものを作り出す「メイカー（Maker）」が集い、展示と交流を行います。
- その他: 東京など全国で開催されています。京都は2019年が初開催で、今回は4年ぶりの2回目です。
"""
INSTRUCTIONS_PROMPT = """
userからの入力に対して以下の制約に従って返答してください。
。#### 規約 ####
-「まゆまろくん？」と呼ばれた場合はあいさつをしてください
- あなたは、Maker Faire Kyoto 2023 の案内役。名前は、まゆまろです。
- userとの会話を通して、[作品データ]の中から、適する作品を探してください。カテゴリ、作者、タイトル、紹介文などの情報から作品を探してください。タイトルや作者は、曖昧な単語で検索してください。
- カテゴリ：エレクトロニクス、IoT、AI、VR／AR／MR、ロボティクス、モビリティ、ドローン、デジタルファブリケーション、FabLab、アシスティブテクノロジー、クラフト、デザイン／アート、ファッション、ミュージック／サウンド、ゲーミング／トイ、教育、キッズ & ファミリー、フード、サイエンス、宇宙、バイオ、企業内の部活動、地方からの出展、Maker Pro、Young Makers
- 話の流れを理解し、場合によっては前におすすめした作品を参照してください。。
ーカテゴリーを尋ねるユーザはカテゴリー名を紹介してください
- 一度に紹介する作品は、最大で1つまでとします。
- 返答文の長さは、2~3文を目安に、必ず100文字程度にしてください。

"""
#- 初めてのユーザはカテゴリーを紹介してください。

# テンプレートを用いてプロンプトを作成
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])
# AIとの会話を管理
# 会話の履歴を保持するためのオブジェクト
#memory = ConversationBufferMemory(return_messages=True)
llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo")
#conversation = ConversationChain(llm=chat, memory=memory, prompt=prompt,verbose=True)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm,k=2)
)


# ユーザからの入力を受け取り、変数commandに代入ge-muwo
command = input("You: ")

while True:
  actiom_resonse,action_score=_dispatch_action(command,'data3')
  if(action_score<0.3718):
    response = conversation.predict(input=actiom_resonse[0]+INSTRUCTIONS_PROMPT)


  else:
      response = conversation.predict(input=actiom_resonse+INSTRUCTIONS_PROMPT)
  print(f"AI: {response}")
  command = input("You: ")
  if command == "next":
      response = conversation.predict(input=actiom_resonse[1])
      print(f"AI: {response}")

  if command == "exit":
      break





