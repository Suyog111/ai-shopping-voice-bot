from __future__ import annotations
import operator
from typing import TypedDict, Annotated, Literal, List
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

def agent_result_reducer(current: list[dict], update: list[dict]) -> list[dict]:
     """link operator.add, but an empty list signals a reset."""
     if not update:
          return []
     return current+update

class AgentTask(BaseModel):
     """A single task assigned to specialist agent."""
     agent: Literal["product_agent", "support_agent"] = Field(
          description="which agent handles this task"
     )
     task_description: str = Field(
          description="what agent should do"
     )

class ClassficaitonResult(BaseModel):
     """ Orchestator's routing decision. """
     tasks: List[AgentTask] = Field(desciption="Task to dispatch")
     requires_synthesis: bool = Field(
          description="True when multiple agent must have their result merged"
     )
     reasoning: str = Field("Brief explanation of rouing decision") 

class AxiomCartState(TypedDict):
     """Top-level state that flows through the entire graph."""
     # Conversation
     messages: Annotated[list[AnyMessage], operator.add]
     user_query: str

     # Routing
     tasks: list[AgentTask]
     require_synthesis: bool

     agent_results: Annotated[list[dict], agent_result_reducer]

     final_answer: str

class WorkerInput(TypedDict):
    """Payload delivered to an agent worker node via Send().

    NOTE: This is intentionally flat — no nested Pydantic objects —
    so that Send() serialisation works without surprises.
    """

    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    task_description: str



       

