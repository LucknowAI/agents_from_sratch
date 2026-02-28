# =============================================================================
# Travel Planning Agent — Built with Pydantic AI
# =============================================================================
#
# HOW IT WORKS:
#   Pydantic AI is the most minimal of the three approaches.
#   We just:
#     1. Create an Agent with a model name and a system prompt.
#     2. Register a tool using the @agent.tool_plain decorator.
#     3. Call agent.run_sync() — it handles everything else.
#
# The @agent.tool_plain decorator reads the type hints to build the
# tool schema automatically — no docstring parsing needed.
#
# INSTALL:
#   pip install pydantic-ai
# =============================================================================

import os
import requests
from pydantic_ai import Agent

# -----------------------------------------------------------------------------
# STEP 0 — Configuration
# -----------------------------------------------------------------------------

os.environ["GROQ_API_KEY"] = ""  # get one at console.groq.com
os.environ["SERPER_API_KEY"] = ""  # https://serper.dev

agent = Agent(
    "groq:qwen/qwen3-32b",
    system_prompt="You are a helpful travel planning assistant.",
)




@agent.tool_plain
def search(query: str) -> str:
    """Search the web for up-to-date travel information."""

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": os.environ["SERPER_API_KEY"],
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json={"q": query, "num": 5})
    response.raise_for_status()

    results = []
    for item in response.json().get("organic", [])[:5]:
        title   = item.get("title", "")
        snippet = item.get("snippet", "")
        results.append(f"- {title}\n  {snippet}")

    return "\n".join(results) if results else "No results found."

# -----------------------------------------------------------------------------
# STEP 1 — Create the agent
#
# We pass:
#   - the model name in "provider:model" format
#   - a system_prompt that tells the AI how to behave
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# STEP 2 — Register the search tool
#
# @agent.tool_plain turns a regular function into a tool.
# Pydantic AI reads the type hints (query: str) to understand what
# arguments the tool needs — no extra configuration required.
# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------
# STEP 3 — Run it!
#
# run_sync() sends the question to the LLM, calls the tool if needed,
# and returns the final answer — all in one line.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    result = agent.run_sync(
        "Plan a 5-day trip to Japan in April. "
        "What are the best places to visit and tips for cherry blossom season?"
    )
    print("\nAssistant:", result.output)
