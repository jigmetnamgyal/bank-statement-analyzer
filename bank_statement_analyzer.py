import tabula
import tabula.io
import pandas as pd
from langchain_experimental.agents import create_csv_agent

import os

from huggingface_hub import login
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer
import torch

path = "/Users/mac/Desktop/bank-statement-analyzer/content/"

file = path + "bank-statement.pdf"

output = tabula.io.convert_into(
    file,
    "/Users/mac/Desktop/bank-statement-analyzer/content/converted.csv",
    output_format="csv",
    lattice=True,
    stream=False,
    pages="all",
)

df = pd.read_csv(path + "converted.csv")
df.head()

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

# ! pip install tabulate
agent = create_csv_agent(
    model,
    "/Users/mac/Desktop/bank-statement-analyzer/content/converted.csv",
    verbose=True,
    allow_dangerous_code=True,
)

agent.run("how many rows are there?")
