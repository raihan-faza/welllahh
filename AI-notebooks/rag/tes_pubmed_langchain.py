
from InstructorEmbedding import INSTRUCTOR

from langchain_community.embeddings import HuggingFaceInstructEmbeddings


from langchain_chroma import Chroma
from dspy.retrieve.chromadb_rm import ChromadbRM
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from langchain.docstore.document import Document

import sys
import os
import dspy
from metapub import PubMedFetcher
import google.generativeai as genai



# jelek context yang dihasilin pubmed langchain

load_dotenv()


os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')



model_name = "hkunlp/instructor-xl"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


class GeminiLM(dspy.LM):
    def __init__(self, model, api_key=None, endpoint=None, **kwargs):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"] or api_key)

        self.endpoint = endpoint
        self.history = []

        super().__init__(model, **kwargs)
        self.model = genai.GenerativeModel(model)

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Custom chat model working for text completion model
        prompt = '\n\n'.join([x['content'] for x in messages] + ['BEGIN RESPONSE:'])

        completions = self.model.generate_content(prompt)
        self.history.append({"prompt": prompt, "completions": completions})

        # Must return a list of strings
        return [completions.candidates[0].content.parts[0].text]
    
    

vector_store = Chroma(
    collection_name="welllahh_rag_collection_chromadb",
    embedding_function=instructor_embeddings,
    persist_directory="./chroma_langchain_db2",  # Where to save data locally, remove if not necessary
)

pubmed_fetcher = PubMedFetcher()

def index_pubmed_docs_based_on_query(query):
    # loader = PubMedLoader(query) & PubmedQueryRun # hasilnya jelek
    # docs = loader.load()
    # vector_store.add_documents(docs[:5], ids=None)
    new_indexed_docs = []
    pmids = pubmed_fetcher.pmids_for_query(query, retmax=10)
    for id in pmids:
        if len(new_indexed_docs) == 2:
            break
        try:
            abstract = pubmed_fetcher.article_by_pmid(id)
            new_doc = Document(
                page_content=abstract.abstract,
            )
            
            new_indexed_docs.append(abstract.xml)
            vector_store.add_documents([new_doc], ids=None, add_to_docstore=True)
        except Exception as e:
            print(e)
            continue


## DSPY
ef = embedding_functions.InstructorEmbeddingFunction(model_name="hkunlp/instructor-xl", device="cpu")

retriever_model = ChromadbRM(
    'welllahh_rag_collection_chromadb',
    './chroma_langchain_db2',
    embedding_function=ef,
    k=3
)


llm = dspy.LM(model='gemini/gemini-1.5-flash', temperature=0)

dspy.settings.configure(lm=llm, rm=retriever_model)



class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    # answer = dspy.OutputField(desc="If the answer is not in the context, answer based on your knowledge") # gakbisa
    # answer = dspy.OutputField(desc="If it is not in the context, answer based on your knowledge")
    #If it is not in context, answer according to your knowledge and add the correct answer that is taken from the context text.
    answer = dspy.OutputField(desc="If it is not in context, answer according to your knowledge and add answers taken from the context text. Explain your answer in detail.")
    # answer = dspy.OutputField()

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


from dsp.utils import deduplicate
qa = dspy.Predict('question: str -> response: str')

class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=5, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        question = qa(question=question+ "; translate to English").response
        # index_pubmed_docs_based_on_query(question)
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        translate_answer = qa(question=pred.answer+ "; translate to Indonesian").response

        # return dspy.Prediction(context=context, answer=pred.answer)
        return dspy.Prediction(context=context, answer=translate_answer)




my_question = "Apa gejala-gejala Diabetes mellitus? dan bagaimana cara mengobatinya?"

uncompiled_baleen = SimplifiedBaleen()  # uncompiled (i.e., zero-shot) program
pred = uncompiled_baleen(my_question)



print(pred)