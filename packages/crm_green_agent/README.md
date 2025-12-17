# CRMArena Green Agent

An A2A-based evaluation agent for testing other agents on CRM tasks using the CRMArena benchmark.

## Overview

This agent acts as an evaluator (green agent) that:
1. Receives evaluation requests with a white agent URL and task configuration
2. Presents CRM tasks to the white agent via A2A protocol
3. Evaluates the white agent's responses using CRMArena's evaluation logic
4. Returns success metrics and performance data

## Architecture

Based on the Tau Bench green agent pattern, adapted for CRMArena:

```
┌─────────────────┐          ┌─────────────────┐
│  CRM Green      │  A2A     │  White Agent    │
│  Agent          │◄────────►│  (under test)   │
│  (Evaluator)    │          │                 │
└────────┬────────┘          └─────────────────┘
         │
         ▼
┌─────────────────┐
│  CRMArena Env   │
│  (Salesforce)   │
└─────────────────┘
```

## Installation

```bash
# Install dependencies
uv sync

# Set up environment variables
touch .env
# Edit .env and add your GOOGLE_API_KEY and Salesforce credentials
```

## Usage

### Starting the Green Agent

```python
from src.crm_green_agent import start_green_agent

# Configure logging (optional)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
)

# Start the server
start_green_agent(host="localhost", port=9001)
```

Or from command line:

```bash
# Option 1
python -m src.crm_green_agent.agent

# Option 2
python -m src.crm_green_agent

# Option 3
uv build
crm_green_agent

# Option 4 (pretty much option 3)
uv run crm-green-agent
```

### Logging Configuration

This section only applies if using the agent programmatically, not if using the agent from the command line.

The agent uses Python's logging module. Make sure to configure logging before using the agent. You can configure logging in several ways:

```python
import logging

# Minimal setup
logging.basicConfig()

# Suggested: Basic setup with good formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
)

# Advanced: Write logs to file and console with different levels
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

# Get a logger for your module
logger = logging.getLogger(__name__)
```

### Sending an Evaluation Request

Send a message to the green agent with the following format:

```
<white_agent_url>http://localhost:9002</white_agent_url>
<env_config>
{
    "org_type": "b2b",
    "agent_strategy": "react",
    "task_ids": [0],
    "max_turns": 20
}
</env_config>
```

#### Configuration Options

- **org_type**: `"b2b"`, `"b2c"`, or `"original"` - Selects the task set and Salesforce org
- **agent_strategy**:
  - `"react"` - ReAct strategy with XML-style actions
  - `"act"` - Act-only strategy
  - `"tool_call"` - Native function calling
  - `"tool_call_flex"` - Extended function calling with more tools
- **task_ids**: Array of task indices to evaluate (currently supports single task)
- **max_turns**: Maximum number of interaction turns (default: 20)

### Expected White Agent Behavior

#### For ReAct/Act Strategies

The white agent should respond using XML tags:

```
<execute>SELECT Id, Name FROM Account WHERE Industry = 'Technology'</execute>
```

or

```
<respond>The account ID is 001XX000003GXXX</respond>
```

#### For Tool Call Strategies

The white agent should respond with JSON wrapped in tags:

```
<json>
{
    "name": "query_salesforce",
    "arguments": {
        "query": "SELECT Id, Name FROM Account WHERE Industry = 'Technology'"
    }
}
</json>
```

or

```
<json>
{
    "name": "respond",
    "arguments": {
        "content": "The account ID is 001XX000003GXXX"
    }
}
</json>
```

## Response Format

The green agent will respond with evaluation results:

```
Finished. White agent success: ✅
Metrics: {
  "time_used": 12.5,
  "success": true
}
Info: {
  "end_reason": {...},
  "agent_actions": [...]
}
```

## Supported CRMArena Tasks

The agent supports all CRMArena task categories:

**B2B Tasks:**
- Sales cycle understanding
- Lead routing and qualification
- Opportunity stage management
- Quote approval
- Activity prioritization

**B2C Tasks:**
- Case routing
- Handle time analysis
- Knowledge QA
- Policy violation identification
- Customer trend analysis

**Original CRMArena:**
- All original benchmark tasks

## Development

### Project Structure

```
src/crm_green_agent/
├── __init__.py              # Package exports
├── __main__.py              # Top-level code (e.g. command line app)
├── agent.py                 # Main agent implementation
├── crm_green_agent.toml     # Agent card configuration
└── util.py                  # Shared utilities (parse_tags, A2A client)
```

### Key Components

- **CRMGreenAgentExecutor**: Main executor class handling A2A requests
- **ask_agent_to_solve()**: Coordinates with white agent to solve tasks
- **load_agent_card_toml()**: Loads agent configuration
- **start_green_agent()**: Starts the server

## Troubleshooting

### Common Issues

1. **Import errors from CRMArena**: Make sure the CRMArena submodule is properly initialized
   ```bash
   git submodule update --init --recursive
   ```

2. **Missing environment variables**: Ensure `.env` file has required credentials:
   - `GOOGLE_API_KEY` for LLM access
   - Salesforce credentials (varies by org_type)

3. **Port already in use**: Change the port in `start_green_agent(port=9001)`

## Testing

Basic smoke test:

```bash
# Start the green agent
crm-green-agent

# In another terminal, send a test request using an A2A client
# (Requires a running white agent on port 9002)
```

## License

Follows the same license as CRMArena.
