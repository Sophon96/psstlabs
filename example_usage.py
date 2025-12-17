"""Example usage of the CRM Green Agent."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from crm_green_agent import start_green_agent


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
    print("=" * 60)
    print("CRMArena Green Agent - Evaluation Server")
    print("=" * 60)
    print("\nStarting server on http://localhost:9001")
    print("\nTo send an evaluation request, use the A2A protocol with:")
    print("  - <white_agent_url>: URL of the agent to evaluate")
    print("  - <env_config>: JSON configuration for the task")
    print("\nExample config:")
    print("""
    {
        "org_type": "b2b",
        "agent_strategy": "react",
        "task_ids": [0],
        "max_turns": 20
    }
    """)
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    start_green_agent(host="localhost", port=9001)


if __name__ == "__main__":
    main()
