"""Utility functions for A2A agent communication and message parsing."""

import re
import uuid
from typing import Dict

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (AgentCard, Message, MessageSendParams, Part, Role,
                       SendMessageRequest, SendMessageResponse, TextPart)


def parse_tags(text: str) -> Dict[str, str]:
    """
    Parse XML-like tags from text.

    Example:
        text = "<white_agent_url>http://localhost:9002</white_agent_url>"
        result = parse_tags(text)
        # Returns: {"white_agent_url": "http://localhost:9002"}

    Args:
        text: Text containing XML-like tags

    Returns:
        Dictionary mapping tag names to their content
    """
    tags = {}
    # Find all tags in format <tagname>content</tagname>
    pattern = r"<([^>]+)>(.*?)</\1>"
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        tag_name = match.group(1)
        tag_content = match.group(2).strip()
        tags[tag_name] = tag_content

    return tags


# Initialize A2A client for sending messages to other agents
_a2a_clients = {}


async def send_message(
    url, message, task_id=None, context_id=None
) -> SendMessageResponse:
    if url in _a2a_clients:
        client = _a2a_clients['url']
    else:
        client = A2AClient(httpx_client=httpx.AsyncClient(timeout=120.0), url=url)
        _a2a_clients['url'] = client

    message_id = uuid.uuid4().hex
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            parts=[Part(TextPart(text=message))],
            message_id=message_id,
            task_id=task_id,
            context_id=context_id,
        )
    )
    request_id = uuid.uuid4().hex
    req = SendMessageRequest(id=request_id, params=params)
    response = await client.send_message(request=req)
    return response
