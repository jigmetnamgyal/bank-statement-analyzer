# Semi Structured RAG
from pydantic import BaseModel
from typing import Any

# ! pip install unstructured
# ! pip install open_filename
# ! pip install pdfminer.six
# ! pip install pi_heif
# ! pip install unstructured_inference
from unstructured.partition.pdf import partition_pdf

# ! pip install nltk
import nltk
import os
import ssl

from langchain import hub
from huggingface_hub import login
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer
import torch

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from langchain_core.runnables import RunnablePassthrough

# Download necessary NLTK data files
nltk_data_dir = "/Users/mac/Desktop/bank-statement-analyzer/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)

# Set the NLTK data path
nltk.data.path.append(nltk_data_dir)
ssl._create_default_https_context = ssl._create_unverified_context

# Download necessary NLTK data files
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download("averaged_perceptron_tagger", download_dir=nltk_data_dir)


print(os.listdir(nltk_data_dir))

# !apt-get install poppler-utils
path = "/Users/mac/Desktop/bank-statement-analyzer/content/"

import tabula

file = path + "bank-statement.pdf"

output = tabula.io.convert_into(
    file, "/Users/mac/Desktop/bank-statement-analyzer/content/converted.csv", output_format="csv", lattice=True, stream=False, pages="all"
)

import pandas as pd

df = pd.read_csv(path+"converted.csv")
df.head()

# ! apt-get install tesseract-ocr
# ! pip install pytesseract
# ! pip install unstructured_pytesseract

os.environ["OCR_AGENT"] = "tesseract"

raw_pdf_elements = partition_pdf(
    filename=path + "bank-statement.pdf",
    extract_images_in_pdf=False,
    infer_table_structure=True,
    chunking_strategy="by_title",
    strategy="hi_res",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
)


category_counts = {}

for element in raw_pdf_elements:
    category = str(type(element))
    if category in category_counts:
        category_counts[category] += 1
    else:
        category_counts[category] = 1

unique_categories = set(category_counts.keys())
category_counts


class Element(BaseModel):
    type: str
    text: Any


# Categorize by type
categorized_elements = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        categorized_elements.append(Element(type="table", text=str(element)))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        categorized_elements.append(Element(type="text", text=str(element)))

# Tables
table_elements = [e for e in categorized_elements if e.type == "table"]
print(len(table_elements))

# Text
text_elements = [e for e in categorized_elements if e.type == "text"]
print(len(text_elements))

# ! pip install langchain_core langchain_community
# ! huggingface-cli login

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "bank_statement_analyzer"


obj = hub.pull("rlm/multi-vector-retriever-summarization")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

login(token="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")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B", padding_side="left"
)

model = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-1B",
    task="text-generation",
    device=-1,  # -1 for CPU
    batch_size=2,  # adjust as needed based on GPU map and model size.
    model_kwargs={
        "temperature": 0,
        "max_length": 4096,
        "torch_dtype": torch.bfloat16,
    },
    pipeline_kwargs={"max_new_tokens": 1024},
)

prompt_text = """You are an assistant tasked with summarizing tables and text. \
Give a concise summary of the table or text. Table or text chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

tables = [i.text for i in table_elements]

# table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

texts = [i.text for i in text_elements]
# text_summaries = summarize_chain.batch(texts, {"max_concurrency": 2})


# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="summaries", embedding_function=FastEmbedEmbeddings()
)  # OpenAIEmbeddings())

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(texts)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(tables)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))

# RAG

# Prompt template
template = """Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke("What is the customer name?")
