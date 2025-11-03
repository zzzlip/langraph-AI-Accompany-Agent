from uuid import NAMESPACE_DNS, uuid5

import chromadb
from langchain_core.messages import RemoveMessage
from langgraph.runtime import Runtime
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.ingestion import pipeline, IngestionPipeline
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import typing
import typing_extensions
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
if not hasattr(typing, 'NotRequired'):
    typing.NotRequired = typing_extensions.NotRequired

from base import llm as model
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import summarize_messages, RunningSummary
from state import MemoryState,Context
path = r"" #emmbeding模型
rerank_model_name = r""  # 示例模型
# 注意：对于中文，'BAAI/bge-reranker-base' 或 'BAAI/bge-reranker-large' 通常是更好的选择
# rerank_model_name = "BAAI/bge-reranker-base" # 如果处理中文内容，可以尝试这个
embeddings = HuggingFaceEmbedding(model_name=path)
Settings.embed_model = embeddings
persist_path = "../chroma_db"
client = chromadb.PersistentClient(path=persist_path)
def add_long_memory(text:str,user_id:str):
    collection = client.get_or_create_collection(name=f"memory_{user_id}_collection")
    vector_store = ChromaVectorStore(
        chroma_collection=collection,
        collection_name=f"memory_{user_id}_collection",
    )
    doc_id = str(uuid5(NAMESPACE_DNS, text))
    doc = [Document(text=text,doc_id=doc_id)]
    pipeline = IngestionPipeline(
        transformations=[
            Settings.embed_model,
        ],
        vector_store=vector_store,
        docstore=SimpleDocumentStore()
    )
    pipeline.run(
        documents=doc,  # 确保传入列表
        in_place=True,
        show_progress=True,
    )
    print(collection.count())

def get_full_long_memory(user_id:str):
    collection = client.get_or_create_collection(name=f"memory_{user_id}_collection")
    doc=collection.get(include=['documents'])
    return doc

def get_simility_long_memory(state:MemoryState,user_id)->list[str]:
    collection = client.get_or_create_collection(name=f"memory_{user_id}_collection")
    vector_store = ChromaVectorStore(
        chroma_collection=collection,
        collection_name=f"memory_{user_id}_collection",
    )
    index = VectorStoreIndex.from_vector_store(vector_store,embed_model=Settings.embed_model)
    retriever = index.as_retriever(
        similarity_top_k=10  # 举例：获取最多10个结果
    )
    message=state['short_memory'][-1]
    result=retriever.retrieve(message.content)
    print(len(result))
    print(result)
    rerank = SentenceTransformerRerank(
        model=rerank_model_name,
        top_n=3,
        device='cuda'  # 只保留重排序后的前3个结果
    )




def manage_memory(state:MemoryState,runtime: Runtime[Context]):
    # 自定义标签生成提示词
    user_id = runtime.context.user_id
    messages = state["short_memory"][:-1]
    long_memory = get_full_long_memory(user_id)
    system_message = """
# 角色与目标
你是一位专业的记忆块标签生成专家。你的核心任务是深入分析用户提供的"聊天记忆"片段，并为其生成或匹配最精准的"记忆块标签"。你的目标是确保每个标签都能高度概括记忆中的一个核心事件，并且遵循特定的匹配与创建规则。

# 工作流程

1.  **深度分析**：首先，仔细阅读并完全理解`[聊天记忆]`中的所有内容。识别出其中发生的一个或多个核心事件、关键决策或重要信息点。
2.  **匹配优先**：将你分析出的核心事件与`[已有标签列表]`进行逐一比对。如果某个已有标签能够准确、完整地概括记忆中的一个事件，你必须直接沿用该标签。
3.  **按需创建**：如果在`[已有标签列表]`中找不到能够描述某个核心事件的标签，你需要为该事件创建一个全新的标签。
4.  **整合输出**：一个`[聊天记忆]`片段可能包含多个独立的事件。因此，最终的结果应该是一个包含了所有被沿用和新创建标签的列表。

# 规则与约束

1.  **新标签创建规则**：
*   **内容**：必须精准概括事件的核心内容，抓住要点。
*   **细节**：在概括的同时，要包含必要的细节，使其具有区分度。
*   **长度**：绝对不能超过20个汉字。
2.  **行为准则**：
*   **优先复用**：始终优先沿用已有的标签，这是最高指令。
*   **避免冗余**：如果一个已有标签已经覆盖了某个事件，不要再为该事件创建相似的新标签。
*   **多事件处理**：如果记忆中包含多个不相关的核心事件，需要为每个事件都匹配或创建一个标签。
3.  **输出格式**：
*   你的最终输出必须是严格的 JSON 格式。
*   JSON 对象中只包含一个键 `"tags"`。
*   `"tags"` 的值是一个字符串列表 `list[str]`。

# 示例

## 输入
### 聊天记忆:
```
A: 我们下个月去云南的机票订好了吗？
B: 订好了，下周五早上8点的。对了，我看到一个很有意思的咖啡庄园，要不要加到行程里？
A: 好主意！一直想去看看。那就这么定了。
```

### 已有标签列表:
```
["项目A技术方案讨论", "预定下个月去云南的机票", "周末聚餐计划"]
```

## 期望输出
```
{{"tags": ["预定下个月去云南的机票", "云南行程中增加参观咖啡庄园"]}}
```
---

请严格按照上述规则，始终用中文输出标签内容。
"""

    # 创建包含系统指令的提示词模板用于摘要生成
    user_prompt=PromptTemplate.from_template(
        template="""
现在对以下内容进行标签生成：

## 新消息内容:
{messages}

## 已有标签列表:
{existing_summary}

请按照系统提示词中的规则为这些新消息生成或匹配标签。
""",
        partial_variables={'existing_summary':long_memory},

    )
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        user_prompt
    ])
    # 使用自定义提示词进行摘要
    summarization_result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        max_tokens=512,
        max_tokens_before_summary=50,
        max_summary_tokens=50,
        initial_summary_prompt=summary_prompt
    )
    if summarization_result.running_summary:
        memory=summarization_result.running_summary.summary
        prase=JsonOutputParser()
        memory=prase.parse(memory)
        tags=memory.tags
        for tag in tags:
            add_long_memory(tag,user_id)
        return {"short_memory": [RemoveMessage(id=m.id) for m in messages[:-1]]}

