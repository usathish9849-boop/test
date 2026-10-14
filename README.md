#!/usr/bin/env python3
"""A command-line research agent built on the Claude API.

It can search the web (Anthropic's server-side web search tool), read pages
in full, and save a finished report to a markdown file.

Usage:
    export ANTHROPIC_API_KEY=your-key
    python agent.py
"""

import os
import re
import sys
from datetime import datetime

import anthropic
import requests

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 4096
MAX_STEPS = 15  # safety cap on agent loop iterations per question
REPORT_DIR = "reports"

SYSTEM_PROMPT = """You are a careful research assistant.

For each question:
1. Search the web for several distinct angles rather than relying on one source.
2. Use fetch_url to read the most promising pages in full when snippets are not enough.
3. Cross-check important claims; say so when sources disagree or evidence is thin.
4. Write a clear answer in your own words, ending with a 'Sources:' list of URLs.
5. Only call save_report if the user asks you to save or write up a report.
"""

TOOLS = [
    # Server-side tool: Anthropic runs the search for you.
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 6},
    # Client-side tools: this script runs them.
    {
        "name": "fetch_url",
        "description": "Download a web page and return its readable text (truncated to ~12,000 characters).",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full http(s) URL"}},
            "required": ["url"],
        },
    },
    {
        "name": "save_report",
        "description": "Save a finished markdown report to disk and return the file path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "markdown": {"type": "string", "description": "Full report body in markdown"},
            },
            "required": ["title", "markdown"],
        },
    },
]


def fetch_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "research-agent/1.0"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"Error fetching {url}: {exc}"
    text = resp.text
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:12000] or "Page had no readable text."


def save_report(title: str, markdown: str) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:50] or "report"
    path = os.path.join(REPORT_DIR, f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{markdown}\n")
    return f"Saved report to {path}"


def run_tool(name: str, args: dict) -> str:
    if name == "fetch_url":
        return fetch_url(args["url"])
    if name == "save_report":
        return save_report(args["title"], args["markdown"])
    return f"Unknown tool: {name}"


def run_agent(client: anthropic.Anthropic, messages: list) -> None:
    """Run the agent loop until Claude gives a final answer."""
    for _ in range(MAX_STEPS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(block.text)
            elif block.type == "tool_use":
                print(f"  [tool] {block.name}({_short(block.input)})")
            elif block.type == "server_tool_use":
                print(f"  [search] {_short(block.input)}")

        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": run_tool(block.name, block.input),
                        }
                    )
            messages.append({"role": "user", "content": results})
        elif response.stop_reason == "pause_turn":
            continue  # server-side tool still working; send the turn back as-is
        else:
            return
    print("\n(Stopped: reached the step limit for this question.)")


def _short(obj, limit: int = 80) -> str:
    s = str(obj)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first (see README.md).")
    client = anthropic.Anthropic()
    messages: list = []
    print("Research agent ready. Ask a question, or type 'quit' to exit.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        messages.append({"role": "user", "content": question})
        print()
        try:
            run_agent(client, messages)
        except anthropic.APIError as exc:
            print(f"API error: {exc}")
            messages.pop()  # drop the failed turn so the next question starts clean
        print()


if __name__ == "__main__":
    main()# test
