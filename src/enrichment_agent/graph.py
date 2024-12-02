"""Define a data enrichment agent.

Works with a chat model with tool calling support.
"""

import json
from typing import Any, Dict, List, Literal, Optional, Union, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from enrichment_agent import prompts
from enrichment_agent.configuration import Configuration
from enrichment_agent.state import InputState, OutputState, State
from enrichment_agent.tools import scrape_website, search
from enrichment_agent.utils import init_model


async def generate_subtopics_agent(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Generate subtopics for the given research topic.

    This node runs before the main research to break down the topic into subtopics.
    """
    # Define the subtopics extraction schema
    subtopics_schema = {
        "type": "object",
        "properties": {
            "subtopics": {
                "type": "array",
                "description": "List of 10 action-oriented subtopics",
                "items": {
                    "type": "string",
                    "description": "A clear, action-oriented subtopic that starts with a verb"
                },
                "minItems": 10,
                "maxItems": 10
            }
        },
        "required": ["subtopics"]
    }

    # Define the 'Info' tool for subtopics
    info_tool = {
        "name": "Info",
        "description": "Call this when you have generated all 10 subtopics",
        "parameters": subtopics_schema,
    }

    # Format the subtopics generation prompt with the topic
    p = prompts.GENERATE_SUBTOPICS_PROMPT.format(topic=state.topic)

    # Create the messages list with the formatted prompt
    messages = [HumanMessage(content=p)]

    # Initialize the raw model with the provided configuration and bind the tools
    raw_model = init_model(config)
    model = raw_model.bind_tools([scrape_website, search, info_tool], tool_choice="any")
    response = cast(AIMessage, await model.ainvoke(messages))

    # Initialize info to None
    info = None

    # Check if the response has tool calls
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "Info":
                info = tool_call["args"]
                break

    response_messages: List[BaseMessage] = [response]
    if not response.tool_calls:  # If LLM didn't respect the tool_choice
        response_messages.append(
            HumanMessage(content="Please respond by calling one of the provided tools.")
        )

    return {
        "messages": response_messages,
        "subtopics": info["subtopics"] if info else None,
        "loop_step": 0,  # Reset loop step as this is the start
    }


async def call_agent_model(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Call the primary Language Model (LLM) to decide on the next research action.

    This asynchronous function performs the following steps:
    1. Initializes configuration and sets up the 'Info' tool, which is the user-defined extraction schema.
    2. Prepares the prompt and message history for the LLM.
    3. Initializes and configures the LLM with available tools.
    4. Invokes the LLM and processes its response.
    5. Handles the LLM's decision to either continue research or submit final info.
    """
    # Load configuration from the provided RunnableConfig
    configuration = Configuration.from_runnable_config(config)

    # Define the 'Info' tool, which is the user-defined extraction schema
    info_tool = {
        "name": "Info",
        "description": "Call this when you have gathered all the relevant info",
        "parameters": state.extraction_schema,
    }

    # Format the prompt defined in prompts.py with the extraction schema and topic
    p = configuration.prompt.format(
        info=json.dumps(state.extraction_schema, indent=2), topic=state.topic
    )

    # Create the messages list with the formatted prompt and the previous messages
    messages = [HumanMessage(content=p)] + state.messages

    # Initialize the raw model with the provided configuration and bind the tools
    raw_model = init_model(config)
    model = raw_model.bind_tools([scrape_website, search, info_tool], tool_choice="any")
    response = cast(AIMessage, await model.ainvoke(messages))

    # Initialize info to None
    info = None

    # Check if the response has tool calls
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "Info":
                info = tool_call["args"]
                break
    if info is not None:
        # The agent is submitting their answer;
        # ensure it isn't erroneously attempting to simultaneously perform research
        response.tool_calls = [
            next(tc for tc in response.tool_calls if tc["name"] == "Info")
        ]
    response_messages: List[BaseMessage] = [response]
    if not response.tool_calls:  # If LLM didn't respect the tool_choice
        response_messages.append(
            HumanMessage(content="Please respond by calling one of the provided tools.")
        )
    return {
        "messages": response_messages,
        "info": info,
        # Add 1 to the step count
        "loop_step": 1,
    }


class InfoIsSatisfactory(BaseModel):
    """Validate whether the current extracted info is satisfactory and complete."""

    reason: List[str] = Field(
        description="First, provide reasoning for why this is either good or bad as a final result. Must include at least 3 reasons."
    )
    is_satisfactory: bool = Field(
        description="After providing your reasoning, provide a value indicating whether the result is satisfactory. If not, you will continue researching."
    )
    improvement_instructions: Optional[str] = Field(
        description="If the result is not satisfactory, provide clear and specific instructions on what needs to be improved or added to make the information satisfactory."
        " This should include details on missing information, areas that need more depth, or specific aspects to focus on in further research.",
        default=None,
    )


class SubtopicsQuality(BaseModel):
    """Validate whether the generated subtopics are satisfactory."""

    analysis: List[str] = Field(
        description="Provide analysis of the subtopics based on the evaluation criteria (at least 3 points)"
    )
    is_satisfactory: bool = Field(
        description="After providing your analysis, indicate whether the subtopics are satisfactory"
    )
    improvement_suggestions: Optional[str] = Field(
        description="If the subtopics are not satisfactory, provide specific suggestions for improvement",
        default=None,
    )


async def reflect(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Validate the quality of the data enrichment agent's output.

    This asynchronous function performs the following steps:
    1. Prepares the initial prompt using the main prompt template.
    2. Constructs a message history for the model.
    3. Prepares a checker prompt to evaluate the presumed info.
    4. Initializes and configures a language model with structured output.
    5. Invokes the model to assess the quality of the gathered information.
    6. Processes the model's response and determines if the info is satisfactory.
    """
    p = prompts.MAIN_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), topic=state.topic
    )
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"{reflect.__name__} expects the last message in the state to be an AI message with tool calls."
            f" Got: {type(last_message)}"
        )
    messages = [HumanMessage(content=p)] + state.messages[:-1]
    presumed_info = state.info
    checker_prompt = """I am thinking of calling the info tool with the info below. \
Is this good? Give your reasoning as well. \
You can encourage the Assistant to look at specific URLs if that seems relevant, or do more searches.
If you don't think it is good, you should be very specific about what could be improved.

{presumed_info}"""
    p1 = checker_prompt.format(presumed_info=json.dumps(presumed_info or {}, indent=2))
    messages.append(HumanMessage(content=p1))
    raw_model = init_model(config)
    bound_model = raw_model.with_structured_output(InfoIsSatisfactory)
    response = cast(InfoIsSatisfactory, await bound_model.ainvoke(messages))
    if response.is_satisfactory and presumed_info:
        return {
            "info": presumed_info,
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content="\n".join(response.reason),
                    name="Info",
                    additional_kwargs={"artifact": response.model_dump()},
                    status="success",
                )
            ],
        }
    else:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content=f"Unsatisfactory response:\n{response.improvement_instructions}",
                    name="Info",
                    additional_kwargs={"artifact": response.model_dump()},
                    status="error",
                )
            ]
        }


async def reflect_subtopics(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Validate the quality of the generated subtopics.

    This function evaluates whether the subtopics are well-formed,
    comprehensive, and appropriate for the main topic.
    """
    if not state.subtopics:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id="reflect_subtopics",
                    content="No subtopics generated to reflect on",
                    name="Info",
                    status="error",
                )
            ]
        }

    # Format the reflection prompt with the topic and subtopics
    p = prompts.REFLECT_SUBTOPICS_PROMPT.format(
        topic=state.topic,
        subtopics="\n".join(f"- {st}" for st in state.subtopics)
    )

    messages = [HumanMessage(content=p)]

    # Initialize the model with structured output
    raw_model = init_model(config)
    bound_model = raw_model.with_structured_output(SubtopicsQuality)
    response = cast(SubtopicsQuality, await bound_model.ainvoke(messages))

    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"{reflect_subtopics.__name__} expects the last message to be an AI message."
            f" Got: {type(last_message)}"
        )

    if response.is_satisfactory:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content="\n".join(response.analysis),
                    name="Info",
                    additional_kwargs={"artifact": response.model_dump()},
                    status="success",
                )
            ],
        }
    else:
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content=f"Subtopics need improvement:\n{response.improvement_suggestions}",
                    name="Info",
                    additional_kwargs={"artifact": response.model_dump()},
                    status="error",
                )
            ]
        }


def route_after_agent(
    state: State,
) -> Literal["reflect", "tools", "call_agent_model", "__end__"]:
    """Schedule the next node after the agent's action.

    This function determines the next step in the research process based on the
    last message in the state. It handles three main scenarios:

    1. Error recovery: If the last message is unexpectedly not an AIMessage.
    2. Info submission: If the agent has called the "Info" tool to submit findings.
    3. Continued research: If the agent has called any other tool.
    """
    last_message = state.messages[-1]

    # "If for some reason the last message is not an AIMessage (due to a bug or unexpected behavior elsewhere in the code),
    # it ensures the system doesn't crash but instead tries to recover by calling the agent model again.
    if not isinstance(last_message, AIMessage):
        return "call_agent_model"
    # If the "Into" tool was called, then the model provided its extraction output. Reflect on the result
    if last_message.tool_calls and last_message.tool_calls[0]["name"] == "Info":
        return "reflect"
    # The last message is a tool call that is not "Info" (extraction output)
    else:
        return "tools"


def route_after_checker(
    state: State, config: RunnableConfig
) -> Literal["__end__", "call_agent_model"]:
    """Schedule the next node after the checker's evaluation.

    This function determines whether to continue the research process or end it
    based on the checker's evaluation and the current state of the research.
    """
    configurable = Configuration.from_runnable_config(config)
    last_message = state.messages[-1]

    if state.loop_step < configurable.max_loops:
        if not state.info:
            return "call_agent_model"
        if not isinstance(last_message, ToolMessage):
            raise ValueError(
                f"{route_after_checker.__name__} expected a tool messages. Received: {type(last_message)}."
            )
        if last_message.status == "error":
            # Research deemed unsatisfactory
            return "call_agent_model"
        # It's great!
        return "__end__"
    else:
        return "__end__"


def route_after_subtopics(
    state: State,
) -> Literal["call_agent_model", "tools"]:
    """Route to next node after subtopics generation."""
    last_message = state.messages[-1]

    if not isinstance(last_message, AIMessage):
        return "generate_subtopics"

    # If the "Into" tool was called, then the model provided its extraction output. Reflect on the result
    if last_message.tool_calls and last_message.tool_calls[0]["name"] == "Info":
        return "reflect_subtopics"

    # If we have subtopics, move to main research
    if state.subtopics:
        return "reflect_subtopics"
    # Otherwise, let the tools handle the response
    return "tools"


def route_after_subtopics_reflection(
    state: State,
) -> Literal["generate_subtopics", "call_agent_model"]:
    """Route to next node after subtopics reflection."""
    last_message = state.messages[-1]

    if not isinstance(last_message, ToolMessage):
        return "generate_subtopics"

    # If subtopics were satisfactory, move to main research
    if last_message.status == "success":
        return "call_agent_model"
    # Otherwise, regenerate subtopics
    return "generate_subtopics"


# Create the graph
workflow = StateGraph(
    State, input=InputState, output=OutputState, config_schema=Configuration
)
workflow.add_node("generate_subtopics", generate_subtopics_agent)
workflow.add_node("reflect_subtopics", reflect_subtopics)
# workflow.add_node("call_agent_model", call_agent_model)
# workflow.add_node("reflect", reflect)
workflow.add_node("tools", ToolNode([search, scrape_website]))
workflow.add_edge("__start__", "generate_subtopics")
# workflow.add_edge("tools", "generate_subtopics")
workflow.add_conditional_edges(
    "generate_subtopics",
    route_after_subtopics,
    {
        "reflect_subtopics": "reflect_subtopics",
        "tools": "tools",
        "generate_subtopics": "generate_subtopics"
    }
)
workflow.add_conditional_edges(
    "reflect_subtopics",
    route_after_subtopics_reflection,
    {
        "generate_subtopics": "generate_subtopics",
        "__end__": "__end__"
        # "call_agent_model": "call_agent_model"
    }
)
# workflow.add_conditional_edges("call_agent_model", route_after_agent)

# workflow.add_conditional_edges("reflect", route_after_checker)

graph = workflow.compile()
graph.name = "WritingTopicsResearch"
