"""Utility functions for A2A agent communication and message parsing."""

import re
from typing import Dict
from a2a.client import A2AClient


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
    pattern = r'<([^>]+)>(.*?)</\1>'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        tag_name = match.group(1)
        tag_content = match.group(2).strip()
        tags[tag_name] = tag_content

    return tags


# Initialize A2A client for sending messages to other agents
my_a2a = A2AClient()
