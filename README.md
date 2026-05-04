# Smart Triage AI

Smart Triage AI is a Streamlit-based healthcare assistant that now uses a multi-agent orchestration flow instead of a single ReAct agent.

## Architecture

The current workflow is a LangGraph `StateGraph` with these stages:

- `Supervisor`: routes the request through the graph
- `Intake`: extracts symptoms, duration, and patient context
- `Risk`: scores the case and checks emergency red flags
- `Safety`: builds the final response and validates safety language
- `Human Review`: pauses high-risk cases for doctor approval in the UI

Patient memory is Pinecone-ready for long-term storage and falls back to a local in-memory profile when Pinecone is unavailable or the index is not configured yet.

## Environment

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=smart-triage-patient-memory
PINECONE_DIMENSION=64
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

Notes:

- `GROQ_API_KEY` is optional for the current deterministic graph, but still useful if you keep the legacy ReAct path around.
- If the Pinecone index does not exist, the code attempts to create it with the configured name and region.
- The default vector dimension is `64` because the current implementation uses an internal deterministic embedding for patient profile storage.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Manual Verification

Low-risk path:

- Enter `I have a mild headache and sore throat for 1 day`
- Expected flow: `Supervisor -> Intake -> Risk -> Safety -> Finish`
- Expected result: no doctor approval prompt

High-risk path:

- Enter `I have severe chest pain and shortness of breath since this morning`
- Expected flow: `Supervisor -> Intake -> Risk -> Safety -> Human Review`
- Expected result: UI pauses and shows doctor approval buttons

## Files

- [app.py](app.py)
- [agent/multi_agent.py](agent/multi_agent.py)
- [requirements.txt](requirements.txt)
- [docs/architecture.md](docs/architecture.md)
- [scripts/generate_architecture_diagram.py](scripts/generate_architecture_diagram.py)
