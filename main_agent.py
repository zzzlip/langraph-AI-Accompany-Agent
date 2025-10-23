
from base import llm_google,llm_qwen, llm_kimi, llm
from langgraph.constants import START, END

from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph
from typing import Literal
from generate_content import  generate_talk,generate_diary,generate_dynamic_condition,generate_dynamic_condition_picture,generate_talk_picture
from memory import get_simility_long_memory,manage_memory
from state import MemoryState,Context
def start_talk(state:MemoryState)->dict:
    talk_number=state.get('talk_number',0)
    talk_number=talk_number+1
    print('欢迎开始聊天')
    return {'talk_number':talk_number}

def jude_path(runtime:Runtime[Context])->Literal['optimize_memory','generate_diary','generate_dynamic_condition']:
    return runtime.context.page

def create_main_agent():
        workflow = StateGraph(MemoryState)
        workflow.add_node(start_talk.__name__, start_talk)
        workflow.add_node(generate_diary.__name__, generate_diary)
        workflow.add_node(generate_dynamic_condition.__name__, generate_dynamic_condition)
        workflow.add_node(generate_dynamic_condition_picture.__name__, generate_dynamic_condition_picture)
        workflow.add_node(generate_talk.__name__, generate_talk)
        workflow.add_node(generate_talk_picture.__name__, generate_talk_picture)
        workflow.add_node('get_long_memory',get_simility_long_memory)
        workflow.add_node('optimize_memory',manage_memory)
        workflow.add_edge(START, start_talk.__name__)
        workflow.add_conditional_edges(start_talk.__name__, jude_path)
        workflow.add_edge('optimize_memory', 'get_long_memory')
        workflow.add_edge('get_long_memory', generate_talk.__name__)
        workflow.add_edge(generate_talk.__name__, generate_talk_picture.__name__)
        workflow.add_edge(generate_talk_picture.__name__,END)
        workflow.add_edge(generate_diary.__name__, END)
        workflow.add_edge(generate_dynamic_condition.__name__, generate_dynamic_condition_picture.__name__)
        workflow.add_edge(generate_dynamic_condition_picture.__name__, END)
        checkpointer = MemorySaver()
        agent = workflow.compile(checkpointer=checkpointer)  # Pass checkpointer correctly
        return agent,checkpointer
