# =============================================================================
# Travel Planning Agent — Built with Plain Python
# =============================================================================
#
# HOW IT WORKS (step by step):
#
#  1. We send the user's question to an LLM (AI model).
#  2. The LLM replies with JSON — a structured format that's easy to parse.
#  3. The JSON tells us one of two things:
#       a) "I need to search the web"  → we call the search tool, then ask the
#          LLM again with the search results so it can write a proper answer.
#       b) "I have enough info"        → we just show the reply to the user.
#
# TOOLS AVAILABLE:
#   - search(query) : searches Google via the Serper API
#
# LIBRARIES USED:
#   - litellm   : a simple wrapper to call any LLM (OpenAI, Groq, etc.)
#   - requests  : to make HTTP calls to the Serper search API
# =============================================================================

import os       # to set API keys as environment variables
import json     # to parse the JSON responses from the LLM
import requests # to call the Serper search API
from litellm import completion  # to call the LLM

# -----------------------------------------------------------------------------
# STEP 0 — Configuration
# Put your API keys here before running.
# -----------------------------------------------------------------------------

os.environ["GROQ_API_KEY"] = ""  # get one at console.groq.com
os.environ["SERPER_API_KEY"] = ""  # https://serper.dev

MODEL = "groq/llama-3.3-70b-versatile"   # any model litellm supports

# -----------------------------------------------------------------------------
# STEP 1 — Define the search tool
# This is a plain Python function. We will call it when the LLM asks us to.
# -----------------------------------------------------------------------------

def search(query):
    """Search the web using the Serper API and return the top results."""

    print(f"  Searching for: {query}")

    url = "https://google.serper.dev/search"

    # The API expects an API key in the request headers
    headers = {
        "X-API-KEY": os.environ["SERPER_API_KEY"],
        "Content-Type": "application/json",
    }

    # Send the search query and ask for 5 results
    response = requests.post(url, headers=headers, json={"q": query, "num": 5})
    response.raise_for_status()  # raises an error if the request failed

    # Extract the title and snippet from each result
    results = []
    for item in response.json().get("organic", [])[:5]:
        title   = item.get("title", "")
        snippet = item.get("snippet", "")
        results.append(f"- {title}\n  {snippet}")

    # Join all results into one string and return it
    if results:
        return "\n".join(results)
    else:
        return "No results found."




# -----------------------------------------------------------------------------
# STEP 2 — Write the system prompt
# This tells the LLM how to behave and what format to reply in.
# We ask it to ALWAYS return JSON so our code can easily parse the response.
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a helpful travel planning assistant.

You have access to one tool:
  - search(query): searches the web for up-to-date travel information.

You MUST respond with valid JSON only. No extra text, no markdown.

If you need to search the web to answer, reply with:
{"needs_tool": true, "tool_name": "search", "tool_args": {"query": "your search query"}}

If you already have enough information to answer, reply with:
{"needs_tool": false, "reply": "your full answer here"}
"""

# -----------------------------------------------------------------------------
# STEP 3 — Function to call the LLM
# We send a list of messages (conversation history) and get back a JSON dict.
# -----------------------------------------------------------------------------

def call_llm(messages):
    """Send messages to the LLM and return its response as a Python dict."""

    response = completion(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},  # force the LLM to reply in JSON
    )

    # response.choices[0].message.content is a JSON string — we parse it into a dict
    raw_text = response.choices[0].message.content
    return json.loads(raw_text)

# -----------------------------------------------------------------------------
# STEP 4 — The agent loop
# This is the brain of the agent. It decides whether to use a tool or not.
# -----------------------------------------------------------------------------

def main(user_question):
    """Run the travel planning agent for a given user question."""
    # Build the conversation: system instructions + user's question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_question},
    ]

    llm_response = call_llm(messages)

    if llm_response["needs_tool"] == True:

        tool_name = llm_response["tool_name"]   # e.g. "search"
        tool_args = llm_response["tool_args"]    # e.g. {"query": "Japan travel tips"}

        if tool_name == "search":
            tool_result = search(tool_args["query"])

        print(f"\n[Step 3] Tool result received:")
        print(tool_result)

        # Add the LLM's first response and the tool result to the conversation
        messages.append({"role": "assistant", "content": json.dumps(llm_response)})
        messages.append({
            "role":    "user",
            "content": f"Here are the search results:\n{tool_result}\n\nNow please answer the original question.",
        })
        
        print("\n[Step 4] Asking the LLM for the final answer...")
        llm_response = call_llm(messages)
        print(f"  LLM replied: {llm_response}")

    # Get the final text reply from the JSON
    final_answer = llm_response.get("reply", "No reply generated.")

    print(f"\nAssistant:\n{final_answer}")
    return final_answer


# -----------------------------------------------------------------------------
# STEP 5 — Run it!
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main(
        "Plan a 5-day trip to Japan in April. "
        "What are the best places to visit and tips for cherry blossom season?"
    )
