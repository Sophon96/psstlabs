"""CRM Green Agent implementation - manages CRMArena task assessment and evaluation."""

import json
import logging
import time
import tomllib
from pathlib import Path

import dotenv
import uvicorn

# Configure logging
logger = logging.getLogger(__name__)

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, Message, SendMessageSuccessResponse
from a2a.utils import get_text_parts, new_agent_text_message
from crm_sandbox.agents import ChatAgent, ToolCallAgent
from crm_sandbox.data.assets import (B2B_SCHEMA, B2C_SCHEMA, SCHEMA_ORIGINAL,
                                     TASKS_B2B, TASKS_B2C, TASKS_ORIGINAL)
from crm_sandbox.env import TOOLS, TOOLS_FULL
from crm_sandbox.env.env import ChatEnv, ToolEnv

# Import utilities
from .util import send_message, parse_tags

dotenv.load_dotenv()


def load_agent_card_toml(agent_name):
    """Load agent card configuration from TOML file."""
    current_dir = Path(__file__).parent
    with open(current_dir / f"{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


async def ask_agent_to_solve(white_agent_url, env, task_index, agent_strategy, max_num_steps=30):
    """
    Ask a white agent to solve a CRMArena task.

    Adapted from Tau Bench reference implementation to work with CRMArena environments.

    Args:
        white_agent_url: URL of the white agent to evaluate
        env: CRMArena environment (ChatEnv or ToolEnv)
        task_index: Index of the task to solve
        agent_strategy: Strategy to use (react, act, tool_call, etc.)
        max_num_steps: Maximum number of interaction steps

    Returns:
        Dictionary with reward, info, and total_cost
    """
    total_cost = 0.0

    # Reset environment and get initial observation
    initial_observation, metadata = env.reset(task_index=task_index)
    reward = 0.0
    info = {}

    # Build task description for the white agent
    if agent_strategy in ["react", "act"]:
        # For chat-based strategies, explain the action format
        task_description = f"""
You are a CRM agent working with a Salesforce instance. Your task is to help with the following query:

{initial_observation}

Available actions:
- <execute>SOQL/SOSL query</execute> - Execute a Salesforce query
- <respond>your response</respond> - Provide a response or final answer

Please respond using the specified XML tag format above. After executing queries, analyze the results and provide a final response when ready.

Metadata: {json.dumps(metadata, indent=2)}
"""
    else:
        # For tool calling strategies, provide tools information
        task_description = f"""
You are a CRM agent working with a Salesforce instance. Your task is to help with the following query:

{initial_observation}

Here's a list of tools you can use (you can use at most one tool at a time):
{json.dumps(env.tools_info, indent=2)}

Please respond in the JSON format. Please wrap the JSON part with <json>...</json> tags.
The JSON should contain:
- "name": the tool call function name, or "respond" if you want to respond directly.
- "arguments": the arguments for the tool call, or {{"content": "your message here"}} if you want to respond directly.

Metadata: {json.dumps(metadata, indent=2)}

Next, I'll provide you with tool call results or follow-up information.
"""

    next_green_message = task_description
    context_id = None

    for step in range(max_num_steps):
        ctx_info = f"ctx_id={context_id}" if context_id else "no context"
        logger.info(f"Sending message to white agent (step {step + 1}/{max_num_steps}, {ctx_info})")
        logger.debug(f"Message content: {next_green_message[:200]}..." if len(next_green_message) > 200 else f"Message content: {next_green_message}")

        # Send message to white agent via A2A protocol
        white_agent_response = await send_message(
            white_agent_url, next_green_message, context_id=context_id
        )

        res_root = white_agent_response.root
        assert isinstance(res_root, SendMessageSuccessResponse), "Expected SendMessageSuccessResponse"
        res_result = res_root.result
        assert isinstance(res_result, Message), "Expected Message from white agent"

        if context_id is None:
            context_id = res_result.context_id
        else:
            assert context_id == res_result.context_id, "Context ID should remain the same"

        # Extract text from white agent's response
        text_parts = get_text_parts(res_result.parts)
        assert len(text_parts) == 1, "Expecting exactly one text part from the white agent"
        white_text = text_parts[0]
        logger.info(f"White agent response received")
        logger.debug(f"White agent response:\n{white_text}")

        # Parse the action from white agent's response
        action = None
        if agent_strategy in ["react", "act"]:
            # Parse XML-style tags
            white_tags = parse_tags(white_text)
            if "execute" in white_tags:
                action = {"name": "execute", "content": white_tags["execute"]}
            elif "respond" in white_tags:
                action = {"name": "respond", "content": white_tags["respond"]}
        else:
            # Parse JSON from tool calling response
            white_tags = parse_tags(white_text)
            if "json" in white_tags:
                try:
                    action_dict = json.loads(white_tags["json"])
                    action = {
                        "name": action_dict["name"],
                        "arguments": action_dict.get("arguments", action_dict.get("kwargs", {}))
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from white agent: {e}")
                    action = None

        if action is None:
            logger.warning("Could not parse valid action from white agent response")
            next_green_message = "I couldn't understand your response. Please use the correct format: <execute>query</execute> or <respond>answer</respond>"
            continue

        # Execute the action in the CRMArena environment
        logger.info(f"Executing action: {action['name']}")
        obs, reward, done, step_info = env.step(action)
        info = {**info, **step_info}

        # Prepare next message based on action type and result
        if action["name"] == "execute":
            next_green_message = f"Query execution result:\n{obs}"
        elif action["name"] == "respond":
            next_green_message = f"Thank you for your response."
            if done:
                break
        elif action["name"] in ["query_salesforce", "respond"] and hasattr(env, "tools_dict"):
            # For tool-based environments
            next_green_message = f"Tool call result:\n{obs}"

        if done:
            break

    return {
        "reward": reward,
        "info": info,
        "total_cost": total_cost
    }


class CRMGreenAgentExecutor(AgentExecutor):
    """Agent executor for CRMArena green agent."""

    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute CRMArena evaluation task.

        Expected input format:
        <white_agent_url>http://...</white_agent_url>
        <env_config>
        {
            "org_type": "b2b" | "b2c" | "original",
            "agent_strategy": "react" | "act" | "tool_call" | "tool_call_flex",
            "task_ids": [0],
            "max_turns": 20
        }
        </env_config>
        """
        logger.info("Received evaluation task, parsing request...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)

        white_agent_url = tags.get("white_agent_url")
        env_config_str = tags.get("env_config")

        if not white_agent_url or not env_config_str:
            logger.error("Missing required tags in request")
            await event_queue.enqueue_event(
                new_agent_text_message(
                    "Error: Missing required tags <white_agent_url> or <env_config>"
                )
            )
            return

        env_config = json.loads(env_config_str)
        logger.debug(f"Parsed config: {env_config}")

        # Set up the environment
        logger.info("Setting up CRMArena environment...")
        assert len(env_config["task_ids"]) == 1, "Only single task supported for demo purpose"
        task_index = env_config["task_ids"][0]

        # Select task set and schema based on org_type
        org_type = env_config.get("org_type", "b2b")
        if org_type == "b2b":
            TASKS_NATURAL = TASKS_B2B
            SCHEMA = B2B_SCHEMA
        elif org_type == "b2c":
            TASKS_NATURAL = TASKS_B2C
            SCHEMA = B2C_SCHEMA
        else:
            TASKS_NATURAL = TASKS_ORIGINAL
            SCHEMA = SCHEMA_ORIGINAL

        selected_tasks = {t['idx']: t for t in TASKS_NATURAL} # pyright: ignore[reportArgumentType, reportCallIssue]
        agent_strategy = env_config.get("agent_strategy", "react")
        max_turns = env_config.get("max_turns", 20)

        logger.info(f"Using strategy: {agent_strategy}, org_type: {org_type}, task_index: {task_index}")

        # Create appropriate environment
        if agent_strategy in ["react", "act"]:
            env = ChatEnv(tasks=selected_tasks, org_type=org_type) # pyright: ignore[reportArgumentType]
        elif agent_strategy == "tool_call":
            env = ToolEnv(tools=TOOLS, tasks=selected_tasks, org_type=org_type) # pyright: ignore[reportArgumentType]
        elif agent_strategy == "tool_call_flex":
            env = ToolEnv(tools=TOOLS_FULL, tasks=selected_tasks, org_type=org_type) # pyright: ignore[reportArgumentType]
        else:
            logger.error(f"Unknown agent_strategy: {agent_strategy}")
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: Unknown agent_strategy: {agent_strategy}")
            )
            return

        metrics = {}

        logger.info("Starting evaluation...")
        timestamp_started = time.time()

        try:
            res = await ask_agent_to_solve(
                white_agent_url, env, task_index, agent_strategy, max_num_steps=max_turns
            )

            metrics["time_used"] = time.time() - timestamp_started
            result_bool = metrics["success"] = res["reward"] == 1
            result_emoji = "✅" if result_bool else "❌"

            logger.info(f"Evaluation complete. Success: {result_bool}, Time: {metrics['time_used']:.2f}s")
            await event_queue.enqueue_event(
                new_agent_text_message(
                    f"Finished. White agent success: {result_emoji}\n"
                    f"Metrics: {json.dumps(metrics, indent=2)}\n"
                    f"Info: {json.dumps(res['info'], indent=2)}\n"
                )
            )
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error during evaluation: {str(e)}")
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(agent_name="crm_green_agent", host="localhost", port=9001):
    """Start the CRM green agent server."""
    logger.info(f"Starting CRM green agent server on {host}:{port}")
    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url  # complete all required card fields

    request_handler = DefaultRequestHandler(
        agent_executor=CRMGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    logger.info("Server initialized, starting uvicorn...")
    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    import logging.handlers
    import sys

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel("INFO")
    file_handler = logging.handlers.RotatingFileHandler("agent.log", maxBytes=1048576, backupCount=8)
    file_handler.setLevel("DEBUG")

    logging.basicConfig(
        level=logging.NOTSET,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[console_handler, file_handler]
    )

    start_green_agent()
