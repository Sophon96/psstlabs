#!/usr/bin/env python3
"""top level code"""

import logging
import logging.handlers
import sys

from . import start_green_agent


def main():
    """
    Start the CRM Green Agent server.

    The agent will listen on http://localhost:9001 and accept evaluation requests.

    To test:
    1. Start this green agent: python example_usage.py
    2. Start a white agent on another port (e.g., 9002)
    3. Send an evaluation request to the green agent with:
       <white_agent_url>http://localhost:9002</white_agent_url>
       <env_config>
       {
           "org_type": "b2b",
           "agent_strategy": "react",
           "task_ids": [0],
           "max_turns": 20
       }
       </env_config>
    """
    # Configure logging for the example
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

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("CRMArena Green Agent - Evaluation Server")
    logger.info("=" * 60)
    logger.info("Starting server on http://localhost:9001")
    logger.info("To send an evaluation request, use the A2A protocol with:")
    logger.info("  - <white_agent_url>: URL of the agent to evaluate")
    logger.info("  - <env_config>: JSON configuration for the task")
    logger.info(
        "Example config: {org_type: b2b, agent_strategy: react, task_ids: [0], max_turns: 20}"
    )
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 60)

    start_green_agent(host="localhost", port=9001)


if __name__ == "__main__":
    main()
