import tabula
import tabula.io
import pandas as pd
from langchain_experimental.agents import (
    create_pandas_dataframe_agent,
)
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np
import pandas as pd

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


def clean_currency_column(column):
    cleaned_column = (
        df[column].str.replace("[^\d.-]", "", regex=True).replace("", 0.0).astype(float)
    )
    return cleaned_column


df["Withdrawal"] = clean_currency_column("Withdrawal")
df["Deposit"] = clean_currency_column("Deposit")
df["Balance"] = clean_currency_column("Balance")

headers = df.columns
df = df[~df.apply(lambda row: all(row == headers), axis=1)]

date_pattern = r"^\d{2}-\d{2}-\d{4}$"
df = df[df["Txn. Date"].str.match(date_pattern, na=False)]

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0,
    api_key="AIzaSyDqRUU7S9sVdgjVavXaXAXtYU5_2-QUGWA",
)

df_copy = df.copy()

# ! pip install tabulate
agent_exec = create_pandas_dataframe_agent(
    model,
    df_copy,
    verbose=True,
    allow_dangerous_code=True,
)

agent_exec.run("Do I have more spending or deposit ?")
