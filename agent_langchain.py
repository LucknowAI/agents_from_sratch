# =============================================================================
# Travel Planning Agent — Built with LangChain
# =============================================================================
#
# HOW IT WORKS:
#   LangChain handles the agent loop for us. We just need to:
#     1. Define a tool using the @tool decorator.
#     2. Create an LLM and a prompt.
#     3. Hand everything to AgentExecutor — it manages the loop.
#
# The @tool decorator reads the function's docstring + type hints and
# automatically creates the JSON schema that the LLM uses to call the tool.
#
# INSTALL:
#   pip install langchain langchain-groq
# =============================================================================

import os
import requests
from langchain_groq import ChatGroq                              # Groq LLM wrapper
from langchain.tools import tool                                # decorator to register tools
from langchain.agents import create_agent                       # new unified agent factory

# -----------------------------------------------------------------------------
# STEP 0 — Configuration
# -----------------------------------------------------------------------------

os.environ["GROQ_API_KEY"] = ""  # get one at console.groq.com
os.environ["SERPER_API_KEY"] = ""  # https://serper.dev

# -----------------------------------------------------------------------------
# STEP 1 — Define the search tool
#
# The @tool decorator tells LangChain:
#   "This function is a tool the agent can use."
# LangChain reads the docstring to understand what the tool does.
# The type hints (query: str) tell it what arguments the tool expects.
# -----------------------------------------------------------------------------



@tool
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
# STEP 2 — Set up the LLM
# ChatGroq connects to Groq's fast inference API.
# -----------------------------------------------------------------------------

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

# -----------------------------------------------------------------------------
# STEP 3 — Build the agent
#
# create_agent wires everything together:
#   - model         : the LLM that decides what to do
#   - tools         : functions the LLM can call
#   - system_prompt : standing instructions for the AI
#
# It returns a compiled graph that manages the loop internally
# (call LLM → call tool → call LLM...) — no AgentExecutor needed.
# -----------------------------------------------------------------------------

agent = create_agent(
    llm,
    tools=[search],
    system_prompt="You are a helpful travel planning assistant.",
)

# -----------------------------------------------------------------------------
# STEP 4 — Run it!
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "Plan a 5-day trip to Japan in April. "
                    "What are the best places to visit and tips for cherry blossom season?"
                ),
            }
        ]
    })
    # The final response is the last message in the messages list
    print("\nAssistant:", result["messages"][-1].content)
