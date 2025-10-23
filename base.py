from typing import TypedDict, List, Annotated
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
import api_key


llm_google = ChatOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="gemini-2.5-flash",
    api_key=api_key.google_api,
    temperature=0.8,
    streaming=True,
)
llm_google_pro = ChatOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key.google_api,
    model="gemini-2.5-pro",
    temperature=0.7,
    streaming=True,
)

llm_qwen=ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key.qwen_api,
    model="qwen-max-latest",
    temperature=0.5,
    streaming=True,
)
llm_kimi=ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key.qwen_api,
    model="Moonshot-Kimi-K2-Instruct",
    temperature=0.7,
    streaming=True,
)
llm=ChatOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key.google_api,
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    streaming=True,
)
