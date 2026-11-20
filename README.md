# Text → Numbers: Embeddings (Streamlit Demo)

A beginner-friendly app that shows real tokenizer pieces, token IDs, and a real sentence embedding from Cohere's `embed-v4.0` model.

## Requirements
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- A Cohere API key (get one at https://dashboard.cohere.com/api-keys)
- Internet connection (the model runs on Cohere's servers)

## Run on Windows

1. Open PowerShell in this folder.
2. Create the virtual environment and install packages:

```powershell
uv sync
```

3. Add your Cohere API key. Copy `.env.example` to `.env` and paste your key:

```powershell
Copy-Item .env.example .env
notepad .env
```

(You can also set a `COHERE_API_KEY` environment variable or put it in `.streamlit/secrets.toml`.)

4. Start the app:

```powershell
uv run streamlit run app.py
```

5. Streamlit should open a browser tab, usually at `http://localhost:8501`.

## What to try
- Enter `I love mangoes`, then click **Convert text to numbers**.
- Inspect the token table and embedding vector.
- Expand **Show the complete embedding vector**.
- Compare `I love mangoes` with `I really enjoy eating mangoes`, then compare it with `The train arrives at noon`.

## Important beginner notes
- Token IDs are labels/identifiers; they do not represent the meaning of a word.
- The final sentence embedding is a list of 1536 real numbers for this model.
- The bar chart displays the first 40 dimensions only.
- Each embedding call is billed by Cohere per input token (about $0.12 per 1M tokens for `embed-v4.0`); the app shows the estimated cost of each call.
