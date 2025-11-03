from dataclasses import dataclass
from operator import add

from langgraph.graph import MessagesState
from typing import Annotated, List
from langchain_core.messages import AnyMessage

class MemoryState(MessagesState):
    short_memory:Annotated[List[AnyMessage],'短期记忆',add]
    long_memory:Annotated[List[AnyMessage],'长期记忆']
    character_name:Annotated[str,'人物名称']
    character_profile:Annotated[str,'人物背景介绍']
    diary: Annotated[str, "日记内容"]
    dynamic_condition: Annotated[dict, "朋友圈动态"]
    picture_path: Annotated[str, "聊天图片路径"]
    dynamic_condition_picture_path: Annotated[list[str], "朋友圈动态图片路径"]
    talk_number:int

@dataclass
class Context:
    user_id: str
    page:Annotated[str,'指引工作流']
