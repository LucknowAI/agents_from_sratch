# Agents From Scratch

Three implementations of the same travel planning AI agent, each built with a different approach:

| File | Approach |
|---|---|
| `agent_python.py` | Plain Python — manual agent loop, no framework |
| `agent_langchain.py` | LangChain — framework handles the loop |
| `agent_pydantic_ai.py` | Pydantic AI — minimal, decorator-based |
| `agents_from_scratch.ipynb` | Jupyter notebook walkthrough of all three |

Each agent can search the web (via Serper) and answer travel planning questions.

---

## 1. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Set Up API Keys

You need two API keys:

- **GROQ_API_KEY** — free LLM inference. Get one at [console.groq.com](https://console.groq.com)
- **SERPER_API_KEY** — Google search API. Get one at [serper.dev](https://serper.dev)

Open each Python file and replace the placeholder values near the top (under `STEP 0 — Configuration`):

```python
os.environ["GROQ_API_KEY"]   = "your-groq-api-key"
os.environ["SERPER_API_KEY"] = "your-serper-api-key"
```

---

## 4. Run the Agents

**Plain Python agent:**
```bash
python agent_python.py
```

**LangChain agent:**
```bash
python agent_langchain.py
```

**Pydantic AI agent:**
```bash
python agent_pydantic_ai.py
```
