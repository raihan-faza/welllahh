from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_chroma import Chroma
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from langchain.docstore.document import Document
import sys
import os
import dspy
import google.generativeai as genai
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.load import dumps, loads

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


load_dotenv()


os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


model_name = "hkunlp/instructor-xl"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")


translate_model = genai.GenerativeModel("gemini-1.0-pro")  # buat translate


def translate_text(text, language):
    newText =  text.replace("\n", "  ")
    newText = (
        newText
        + f"; translate to {language}. please just translate the text and don't answer the questions!"
    )
    return translate_model.generate_content(newText).candidates[0].content.parts[0].text


summarizer_model = genai.GenerativeModel("gemini-1.5-flash-8b")


def user_summarizer(text):
    return summarizer_model.generate_content(text).candidates[0].content.parts[0].text


retriever_model = Chroma(
    collection_name="welllahh_rag_collection_chromadb",
    persist_directory="./chroma_langchain_db2",
    embedding_function=instructor_embeddings,
)

retriever = retriever_model.as_retriever(search_kwargs={"k": 8})


# template = """You are an AI language model assistant. Your task is to generate search query of the given user question to retrieve relevant documents from a vector database.  Original question: {question}""" #gabisa malah gak bikin query
template = """Write multiple very short search queries (each queries by a separated by "," & maximum 3 very short search queries) that will help answer complex user questions, make sure in your answer you only give multiple very short search queries (each queries by a separated by "," & maximum 3 very short search queries) and don't include any other text! . Original question: {question}"""
search_query_prompt = ChatPromptTemplate.from_template(template)


class GeminiLLM(LLM):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        # return prompt[: self.n]
        llm = genai.GenerativeModel(
            model_name="tunedModels/gemini-welllahh-zerotemp-lrfv-3536"  # sebelumnya 0
        )  # buat jawab pertanyaan medis

        ans = (
            llm.generate_content(prompt, generation_config={"temperature": 0.05})
            .candidates[0]
            .content.parts[0]
            .text
        )
        return ans

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "gemini-welllahh-zerotemp-lrfv-3536"


llm = GeminiLLM()

generate_query = search_query_prompt | llm | StrOutputParser()

cant_access = {
    "npin.cdc.gov",
    "www.ncbi.nlm.nih.gov",
}  # gak bisa diakses & gak muncul tag <p> nya


def add_websearch_results(query):
    queries = [query]
    if "," in query:
        queries = query.split(",")
    else:
        queries = [query]
    
    websearch_all = []
    
    for queryProc in queries:
        websearch = []
        try:
            results = DDGS().text(queryProc, max_results=8)
        except Exception:
            continue
        for res in results:
            domain = res["href"].split("/")[2]
            if "webmd" in res["href"] or ".pdf" in res["href"] or domain in cant_access:
                continue
            if len(websearch) == 3:
                break
            if ".org" in res["href"] or ".gov" in res["href"] or "who" in res["href"]:

                link = res["href"]
                try:
                    page = requests.get(link).text
                except requests.exceptions.RequestException as errh:
                    print(f"error: {errh}")
                    continue
                doc = BeautifulSoup(page, features="html.parser")
                text = ""
                hs = doc.find_all("h2")
                h3s = doc.find_all("h3")
                ps = doc.find_all("p")
                for h3 in h3s:
                    hs.append(h3)
                for pp in ps:
                    hs.append(pp)

                hs_parents = set()
                for h2 in hs:
                    h2_parent = h2.parent
                    if h2_parent in hs_parents:
                        continue
                    hs_parents.add(h2_parent)
                    h2_adjacent = h2_parent.children
                    for adjacent in h2_adjacent:
                        if adjacent.name == "p" and adjacent.text != "\n":
                            text += adjacent.text + "\n"
                        if (
                            adjacent.name == "h2"
                            or adjacent.name == "h3"
                            or adjacent.name == "h4"
                        ):
                            text += adjacent.text + ": \n"
                        if adjacent.name == "ul" or adjacent.name == "ol":
                            text += ": "
                            for li in adjacent.find_all("li"):
                                text += li.text + ","
                            text += "\n"
                if "Why have I been blocked" in text or text == "" or text == ": \n":
                    continue

                websearch.append(text)
        for resText in websearch:
            websearch_all.append(resText)

    return websearch_all


class DuckDuckGoRetriever(BaseRetriever):
    """A toy retriever that contains the top k documents that contain the user query.

    This retriever only implements the sync method _get_relevant_documents.

    If the retriever were to involve file access or network access, it could benefit
    from a native async implementation of `_aget_relevant_documents`.

    As usual, with Runnables, there's a default async implementation that's provided
    that delegates to the sync implementation running on another thread.
    """

    """List of documents to retrieve from."""
    k: int
    """Number of top results to return"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""
        matching_documents = []
        websearch = add_websearch_results(query)

        for document in websearch:
            if len(matching_documents) > self.k:
                return matching_documents

            matching_documents.append(document)
        return matching_documents


websearch_retriever = DuckDuckGoRetriever(k=4)


def rerank_docs_medcpt(query, docs):
    pairs = [[query, article] for article in docs]
    with torch.no_grad():
        encoded = tokenizer(
            pairs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        )

        logits = model(**encoded).logits.squeeze(dim=1)
        values, indices = torch.sort(logits, descending=True)
        relevant = [docs[i] for i in indices[:6]]
    return relevant


retrieval_chain = generate_query | {
    "chroma": retriever,
    "websearch": websearch_retriever,
    "query": StrOutputParser(),
}


def format_docs(docs):
    query = docs["query"]
    chroma_docs = [doc.metadata["content"] + doc.page_content for doc in docs["chroma"]]
    docs = list(set(chroma_docs + docs["websearch"]))
    relevant_docs = rerank_docs_medcpt(query, docs)
    context = "\n\n".join(doc for doc in relevant_docs)
    return context


template = """Answer the question based only on the following context:
{context}
\n

please do not mention the same answer more than once and Don't say 'the given text does not answer the user's question or is not relevant to the user's question' in your answer. just answer the question don't add any other irrelevant text
\n
Question: {question}
\n
Answer: Let's think step by step.
"""


prompt = ChatPromptTemplate.from_template(template)


# answer_chain = (
#     {"context": retrieval_chain | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | {"llm_output": StrOutputParser(), "context": retrieval_chain | format_docs}
# )
answer_chain = (
    # {"context": retrieval_chain | format_docs, "question": RunnablePassthrough()}
     prompt
    | llm
    | {"llm_output": StrOutputParser()}
)


# def answer(question):

#     # answer = answer_chain.invoke(input=question, )
#     docs = retrieval_chain
#     context = format_docs(docs)
#     answer = answer_chain.invoke({"context": context, "question": question})
#     return answer

def answer(question):

    # answer = answer_chain.invoke(input=question, )
    docs = retrieval_chain.invoke(question)
    context = format_docs(docs)
    answer = answer_chain.invoke({"context": context, "question": question})
    return answer, context


def answer_pipeline(question, chat_history, riwayat_penyakit):
    user_context = ""
    if chat_history != "":
        user_context = user_summarizer(
            text=chat_history
            + "; summarize the user's health condition based on the user's chat history above! only explain the user's health condition and nothing else!"
        )

    new_question = question
    if user_context != "" and "insufficient data" not in user_context.lower():
        new_question = ".my health condition: " + user_context + ". User Question: " + question
    if riwayat_penyakit != "":
        new_question = "my medical history: " + riwayat_penyakit + ". User Question: " + new_question

    new_question = new_question.split("\n")
    new_question = " ".join(new_question)
    question = translate_text(new_question, "English")
    question = question.replace("\n", "  ")
    print("retrieving relevant passages and answering user question....")
    # pred = answer(question)
    pred, context = answer(question)
    translate_answer = translate_text(pred["llm_output"], "Indonesian")
    return translate_answer, context
