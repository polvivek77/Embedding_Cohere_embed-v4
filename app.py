import logging
import os
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import cohere
import requests
from dotenv import load_dotenv
from tokenizers import Tokenizer


MODEL_NAME = "embed-v4.0"
MODEL_URL = "https://docs.cohere.com/docs/cohere-embed"
MODEL_LICENSE = "Proprietary (Cohere API terms)"
EMBED_DIMENSION = 1536  # embed-v4.0 supports 256, 512, 1024 or 1536; 1536 is the default.
EMBED_INPUT_TYPE = "search_document"
# Cohere list price for embed-v4.0 text input. Check https://cohere.com/pricing if it changes.
PRICE_PER_MILLION_TOKENS = 0.12
COST_HELP = (
    f"This model runs on Cohere's servers and is billed per input token "
    f"(${PRICE_PER_MILLION_TOKENS:.2f} per 1M tokens for {MODEL_NAME}). "
    "Costs shown here are estimates from the tokens Cohere reports as billed."
)
APP_DIR = Path(__file__).resolve().parent
LOG_FILE = APP_DIR / "app.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | session=%(session_id)s | %(name)s:%(lineno)d | %(message)s"

# Lets you keep COHERE_API_KEY in a local .env file instead of a system environment variable.
load_dotenv(APP_DIR / ".env")


class _SessionIdDefault(logging.Filter):
    # Library log records have no session id; give them a placeholder so the format string works.
    def filter(self, record):
        if not hasattr(record, "session_id"):
            record.session_id = "-"
        return True


def setup_logging():
    # Streamlit re-executes this script on every interaction, so only attach handlers once per process.
    logger = logging.getLogger("embedding_app")
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_SessionIdDefault())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_SessionIdDefault())

    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    # Record the Cohere SDK's own messages in app.log too.
    cohere_logger = logging.getLogger("cohere")
    cohere_logger.setLevel(logging.INFO)
    cohere_logger.addHandler(file_handler)

    logger.info("Logging initialised; writing to %s", LOG_FILE)
    return logger


logger = setup_logging()

st.set_page_config(page_title="Text → Numbers: Embeddings", page_icon="🧠", layout="wide")

is_new_session = "session_id" not in st.session_state
if is_new_session:
    st.session_state["session_id"] = uuid.uuid4().hex[:8]
log = logging.LoggerAdapter(logger, {"session_id": st.session_state["session_id"]})
if is_new_session:
    log.info("New browser session started")
log.debug("Script run started | converted=%s", st.session_state.get("converted", False))


def get_api_key():
    # Order: environment variable / .env file, then Streamlit secrets (.streamlit/secrets.toml).
    key = os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("COHERE_API_KEY")
    except Exception:
        return None


class CohereEmbedder:
    """Small wrapper so the rest of the app can ask for tokens and embeddings without knowing about the API."""

    def __init__(self, client, tokenizer, max_seq_length):
        self.client = client
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def tokenize(self, text):
        # Cohere's embed endpoint adds no visible special tokens, so the table shows the raw text pieces.
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        return encoding.tokens[: self.max_seq_length], encoding.ids[: self.max_seq_length]

    def encode(self, texts):
        # Returns (vectors, billed_input_tokens). Accepts one string or a list of strings.
        single = isinstance(texts, str)
        response = self.client.embed(
            model=MODEL_NAME,
            texts=[texts] if single else list(texts),
            input_type=EMBED_INPUT_TYPE,
            embedding_types=["float"],
            output_dimension=EMBED_DIMENSION,
            truncate="END",
        )
        vectors = np.asarray(response.embeddings.float_, dtype=float)
        billed = response.meta.billed_units if response.meta else None
        billed_tokens = int((billed.input_tokens if billed else None) or 0)
        return (vectors[0] if single else vectors), billed_tokens


def token_cost(billed_tokens):
    return billed_tokens * PRICE_PER_MILLION_TOKENS / 1_000_000


def format_cost(cost):
    # Single sentences cost fractions of a cent, so show enough decimals to see a non-zero value.
    return f"${cost:.10f}".rstrip("0") if cost else "$0.00"


@st.cache_resource
def load_model():
    # Connects to Cohere and downloads the model's tokenizer once. Returns the embedder plus its specs for the model card.
    logger.info("Connecting to Cohere model '%s'", MODEL_NAME)
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "COHERE_API_KEY is not set. Add it to a .env file, your environment, or .streamlit/secrets.toml."
        )

    start = time.perf_counter()
    client = cohere.ClientV2(api_key=api_key)
    info = client.models.get(MODEL_NAME)
    if not info.tokenizer_url:
        raise RuntimeError(f"Cohere did not return a tokenizer for {MODEL_NAME}")
    tokenizer_response = requests.get(info.tokenizer_url, timeout=60)
    tokenizer_response.raise_for_status()
    tokenizer = Tokenizer.from_str(tokenizer_response.text)
    load_seconds = time.perf_counter() - start

    max_seq_length = info.context_length or 128_000
    specs = {
        "dimensions": EMBED_DIMENSION,
        "max_seq_length": int(max_seq_length),
        "endpoints": ", ".join(info.endpoints or []),
        "vocab_size": tokenizer.get_vocab_size(),
        "deprecated": bool(info.is_deprecated),
        "load_seconds": load_seconds,
    }
    logger.info("Model '%s' ready in %.2fs | specs=%s", MODEL_NAME, load_seconds, specs)
    return CohereEmbedder(client, tokenizer, int(max_seq_length)), specs


st.title("🧠 Text → Numbers: See Embeddings in Action")
st.markdown(f":material/memory: **Embedding model in use:** [`{MODEL_NAME}`]({MODEL_URL}) (Cohere)")
st.write(
    "Type text and explore how an embedding model turns it into a list of numbers. "
    "The first run may take a little while because the app connects to Cohere and downloads the tokenizer."
)

with st.expander("🔎 The journey in one glance", expanded=True):
    st.markdown(
        "**Your sentence** → **tokens** → **token IDs** → "
        "**learned vector for each token (model internals)** → **sentence embedding**"
    )
    st.caption(
        "Note: this app shows real token IDs from the model's tokenizer and the real final embedding. "
        "The model's internal token vectors are not the same as token IDs."
    )

default_text = "I love mangoes"
text = st.text_input("✍️ Enter a sentence", value=default_text, max_chars=500)

with st.sidebar:
    st.header(":material/memory: Model in use")
    st.markdown(f"[`{MODEL_NAME}`]({MODEL_URL})")
    try:
        with st.spinner("Loading the model..."):
            _, specs = load_model()
    except Exception as exc:
        log.exception("Failed to load model '%s' for the model card", MODEL_NAME)
        if not get_api_key():
            st.error(f"{exc}")
        else:
            st.error(f"Could not load the model. Details were written to `{LOG_FILE.name}`.")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Dimensions", specs["dimensions"], border=True, help="Length of the vector produced for each sentence.")
        m2.metric("Max input", f"{specs['max_seq_length']:,} tokens", border=True, help="Longer inputs are truncated.")
        m3, m4 = st.columns(2)
        m3.metric("Parameters", "Not disclosed", border=True, help="Cohere does not publish the parameter count.")
        m4.metric("Weights size", "Cloud", border=True, help="The model weights live on Cohere's servers, not this machine.")
        m5, m6 = st.columns(2)
        m5.metric("Cost", f"${PRICE_PER_MILLION_TOKENS:.2f}/1M", border=True, help=COST_HELP)
        m6.metric("Load time", f"{specs['load_seconds']:.1f} s", border=True, help="Time this server took to connect to Cohere and download the tokenizer.")

        details = pd.DataFrame(
            [
                ("Architecture", "Proprietary transformer (Cohere)"),
                ("Output dimensions", "256 · 512 · 1024 · 1536"),
                ("Vocabulary", f"{specs['vocab_size']:,} tokens"),
                ("Input type", EMBED_INPUT_TYPE),
                ("Normalized output", "Yes (unit length)"),
                ("Similarity function", "cosine / dot"),
                ("Embedding type", "float"),
                ("Running on", "Cohere API (cloud)"),
                ("License", MODEL_LICENSE),
                ("Language", "Multilingual (100+ languages)"),
            ],
            columns=["Property", "Value"],
        )
        st.dataframe(details, hide_index=True, alt=f"Technical details of the {MODEL_NAME} model")

if st.button("✨ Convert text to numbers", type="primary"):
    log.info("Convert clicked | text_length=%d", len(text))
    log.debug("Input text: %r", text)
    if not text.strip():
        log.warning("Convert aborted: input text is empty")
        st.warning("Please enter some text first.")
        st.stop()

    try:
        with st.spinner("Loading the model and creating your embedding..."):
            start = time.perf_counter()
            model, specs = load_model()

            # Token IDs are real IDs assigned by this model's tokenizer.
            # They are capped at the model's limit, so the table shows the tokens the model sees.
            tokens, token_ids = model.tokenize(text)
            vector, billed_tokens = model.encode(text)
            elapsed = time.perf_counter() - start
    except Exception:
        log.exception("Failed to create embedding for input text")
        st.error(f"Something went wrong while creating the embedding. Details were written to `{LOG_FILE.name}`.")
        st.stop()

    log.info(
        "Embedding created in %.3fs | tokens=%d | billed_tokens=%d | dimensions=%d | l2_norm=%.4f",
        elapsed, len(tokens), billed_tokens, len(vector), float(np.linalg.norm(vector)),
    )
    log.debug("Tokens: %s", tokens)
    log.debug("Token IDs: %s", token_ids)
    log.debug("Embedding (first 10 values): %s", np.round(vector[:10], 6).tolist())

    st.session_state["tokens"] = tokens
    st.session_state["token_ids"] = token_ids
    st.session_state["vector"] = vector
    st.session_state["embed_seconds"] = elapsed
    st.session_state["embed_cost"] = token_cost(billed_tokens)
    st.session_state["max_tokens"] = specs["max_seq_length"]
    st.session_state["converted_text"] = text
    st.session_state["converted"] = True

if st.session_state.get("converted", False):
    tokens = st.session_state["tokens"]
    token_ids = st.session_state["token_ids"]
    vector = st.session_state["vector"]
    st.success(f"Done! Embedding created with `{MODEL_NAME}` for: “{st.session_state['converted_text']}”")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Embedding dimensions", len(vector), border=True)
    s2.metric(
        "Tokens used", f"{len(tokens)} / {st.session_state['max_tokens']:,}", border=True,
        help="Tokens in your sentence, as counted by the model's tokenizer, out of the model's maximum.",
    )
    s3.metric(
        "Time taken", f"{st.session_state['embed_seconds'] * 1000:.0f} ms", border=True,
        help="Tokenization plus the Cohere API round trip for this sentence. The first run also includes connecting to Cohere.",
    )
    s4.metric("Cost", format_cost(st.session_state.get("embed_cost", 0.0)), border=True, help=COST_HELP)

    st.subheader("1️⃣ Tokenization + Token IDs")
    st.write("The tokenizer splits the text into pieces. Each piece gets a numeric ID.")
    token_df = pd.DataFrame({
        "Token (text piece)": tokens,
        "Token ID": token_ids
    })
    st.dataframe(token_df, width="stretch", hide_index=True)
    st.caption("Token pieces may be whole words, parts of words, punctuation, or special markers.")

    st.subheader("2️⃣ The final sentence embedding")
    st.write(
        f"The model `{MODEL_NAME}` represents your entire sentence using **{len(vector)} numbers**. "
        "These numbers are learned by the model during training."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.code("[" + ", ".join(f"{x:.4f}" for x in vector[:12]) + (", ...]" if len(vector) > 12 else "]"))
        st.caption("Showing the first 12 values above; the full vector is available below.")
    with col2:
        preview_count = min(40, len(vector))
        chart_df = pd.DataFrame({
            "Dimension": [f"n{i}" for i in range(preview_count)],
            "Value": vector[:preview_count]
        }).set_index("Dimension")
        st.bar_chart(chart_df)
        st.caption(f"Visualization of the first {preview_count} dimensions. Each dimension is one position in the vector.")

    with st.expander("📋 Show the complete embedding vector"):
        st.code("[" + ", ".join(f"{x:.6f}" for x in vector) + "]")
        if st.download_button(
            "⬇️ Download embedding as CSV",
            data=pd.DataFrame({"dimension": range(len(vector)), "value": vector}).to_csv(index=False),
            file_name="sentence_embedding.csv",
            mime="text/csv"
        ):
            log.info("Embedding CSV downloaded | dimensions=%d", len(vector))

    st.subheader("3️⃣ Compare meanings")
    st.write("Try two sentences. Dot product gives a rough measure of how close their embeddings are.")
    c1, c2 = st.columns(2)
    with c1:
        sentence_a = st.text_input("Sentence A", value=text, key="sentence_a")
    with c2:
        sentence_b = st.text_input("Sentence B", value="I really enjoy eating mangoes", key="sentence_b")

    if st.button("🔗 Compare these sentences"):
        log.info("Compare clicked | len_a=%d | len_b=%d", len(sentence_a), len(sentence_b))
        log.debug("Sentence A: %r | Sentence B: %r", sentence_a, sentence_b)
        if not sentence_a.strip() or not sentence_b.strip():
            log.warning("Compare aborted: one or both sentences are empty")
            st.warning("Enter both sentences to compare.")
        else:
            try:
                start = time.perf_counter()
                # The client is cached, so this skips the connection step. It must be fetched here
                # because this button click is a fresh script run where the Convert block did not execute.
                model, _ = load_model()
                (emb_a, emb_b), billed_tokens = model.encode([sentence_a, sentence_b])
                score = float(np.dot(emb_a, emb_b))
                elapsed = time.perf_counter() - start
            except Exception:
                log.exception("Failed to compare sentences")
                st.error(f"Something went wrong while comparing the sentences. Details were written to `{LOG_FILE.name}`.")
            else:
                log.info("Dot product computed in %.3fs | score=%.4f | billed_tokens=%d", elapsed, score, billed_tokens)
                st.metric("Dot product", f"{score:.3f}")
                st.caption(
                    "Cohere embeddings are unit length, so this dot product equals cosine similarity and falls between −1 and 1. "
                    "There is no universal cutoff: a higher score generally indicates a stronger match; "
                    "interpret it relative to other scores from the same search or comparison."
                )
                st.caption(f"Computed in {elapsed * 1000:.0f} ms · cost {format_cost(token_cost(billed_tokens))} (Cohere API)")

    with st.expander("🧩 What just happened?"):
        st.markdown(
            """
            1. **Tokenizer:** splits the sentence into pieces.
            2. **Token IDs:** gives each piece an ID so the model can process it.
            3. **Embedding model:** uses learned numerical patterns to represent meaning and context.
            4. **Sentence embedding:** produces one fixed-size vector for the whole sentence.

            **Remember:** the token IDs are not the embedding. The final vector is the embedding you commonly store in a vector database for semantic search or RAG.
            """
        )
else:
    st.info("Enter a sentence above and click **Convert text to numbers** to explore the real output.")
    st.markdown("### Example sentences to try")
    st.write("- `I love mangoes`")
    st.write("- `I enjoy eating fruit`")
    st.write("- `The train arrives at noon`")
    st.caption("Similar ideas may have closer vectors, but the exact values depend on the model.")
