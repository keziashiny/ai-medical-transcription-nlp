from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit as st

# ---------- Paths (relative, so they work on Streamlit Cloud) ----------
BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "outputs" / "embeddings" / "nn_index.pkl"
EMB_PATH = BASE_DIR / "outputs" / "embeddings" / "embeddings.npy"
DATA_PATH = BASE_DIR / "outputs" / "processed" / "processed_results.csv"


@st.cache_resource
def load_index_and_model():
    """
    Load nearest neighbors index, sentence-transformer model, embeddings, and dataframe.
    Uses relative paths so it works both locally and on Streamlit Cloud.
    """
    with open(INDEX_PATH, "rb") as f:
        saved = pickle.load(f)

    nn = saved["nn"]
    model_name = saved["model_name"]

    # Load the CSV that matches the embeddings
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    model = SentenceTransformer(model_name)
    embeddings = np.load(EMB_PATH)

    return df, model, nn, embeddings


def semantic_search(query: str, top_k: int = 5) -> pd.DataFrame:
    """Run semantic search over the dataset and return top_k rows."""
    df, model, nn, embeddings = load_index_and_model()

    query_emb = model.encode([query])
    distances, indices = nn.kneighbors(query_emb, n_neighbors=top_k)

    results = df.iloc[indices[0]].copy()
    results["distance"] = distances[0]
    return results


# ------------------- PAGE CONFIG & HEADER -------------------

st.set_page_config(
    page_title="Medical NLP Assistant",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Medical NLP Assistant")
st.markdown(
    """
Explore a medical transcription dataset using **semantic search** and a **chat-style assistant**.  
All answers are generated from the dataset and are **for educational/demo purposes only, not medical advice.**
"""
)

st.divider()

# Tabs for modes
search_tab, chat_tab = st.tabs(["🔍 Search Mode", "💬 Chat Mode"])


# ------------------- 🔍 SEARCH MODE -------------------
with search_tab:
    st.subheader("Semantic Search")

    st.write(
        """
Type a symptom, diagnosis, or clinical description (e.g.  
`"fever and chills"`, `"abdominal pain"`, `"shortness of breath"`, `"postoperative chest pain"`).
        """
    )

    st.sidebar.header("Search Settings")
    top_k_search = st.sidebar.slider(
        "Number of results (search mode)", min_value=3, max_value=15, value=5, step=1
    )

    example_queries = [
        "fever and chills",
        "abdominal pain",
        "shortness of breath",
        "diabetes follow up",
        "postoperative chest pain",
    ]

    st.sidebar.markdown("### Example queries")
    example_search = st.sidebar.selectbox(
        "Try an example:", ["(none)"] + example_queries, key="search_example"
    )

    query_search = st.text_input(
        "Enter a query:",
        value=example_search if example_search != "(none)" else "",
        key="search_input",
    )

    if st.button("Run Search"):
        if not query_search.strip():
            st.warning("Please enter a query first.")
        else:
            with st.spinner("Running semantic search..."):
                results = semantic_search(query_search, top_k=top_k_search)

            st.markdown(f"### 🔍 Top {len(results)} results for: `{query_search}`")

            # Optional specialty filter
            if "medical_specialty" in results.columns:
                specialties = ["All"] + sorted(
                    results["medical_specialty"].dropna().unique().tolist()
                )
                selected_specialty = st.selectbox(
                    "Filter by medical specialty (optional):",
                    specialties,
                    key="search_specialty",
                )

                if selected_specialty != "All":
                    results = results[results["medical_specialty"] == selected_specialty]

            for idx, row in results.iterrows():
                header = row.get("sample_name", f"Sample {idx}")
                with st.expander(f"📄 {header}  |  distance: {row['distance']:.3f}"):

                    if "medical_specialty" in row and not pd.isna(row["medical_specialty"]):
                        st.markdown(f"**Specialty:** {row['medical_specialty']}")

                    if "description" in row and isinstance(row["description"], str):
                        st.markdown(f"**Description:** {row['description']}")

                    if "transcription" in row and isinstance(row["transcription"], str):
                        st.markdown("#### Original Transcription")
                        st.write(row["transcription"])

                    if "clean_text" in row and isinstance(row["clean_text"], str):
                        st.markdown("#### Cleaned Text (Model Input)")
                        st.write(row["clean_text"])


# ------------------- 💬 CHAT MODE -------------------
with chat_tab:
    st.subheader("Chat with the Dataset")

    st.caption(
        "Ask high-level questions like: "
        "`What are common causes of fever?`, "
        "`How is postoperative chest pain described?`, "
        "`What symptoms are associated with shortness of breath?`"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Chat Mode Info")
    st.sidebar.info(
        "In chat mode, your question is turned into a semantic search over the dataset. "
        "The assistant then gives a warm, human-style summary of the most relevant cases. "
        "This is for learning and demo purposes only and is **not medical advice**."
    )

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_query = st.chat_input("Ask something about the cases in this dataset...")
    if user_query:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # --- simple greeting handling ---
        normalized = user_query.lower().strip()
        greeting_keywords = ["hi", "hello", "hey"]

        if normalized in greeting_keywords:
            assistant_reply = (
                "Hi! 👋 It's really nice to meet you.\n\n"
                "I'm your medical NLP assistant. You can ask things like:\n\n"
                "- `What are common causes of fever?`\n"
                "- `How is abdominal pain described?`\n"
                "- `What symptoms are associated with chest pain?`"
            )
            with st.chat_message("assistant"):
                st.markdown(assistant_reply)

            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

        else:
            # Assistant reply based on semantic search
            with st.chat_message("assistant"):
                with st.spinner("Looking for similar cases in the dataset..."):
                    results = semantic_search(user_query, top_k=5)

                    if results.empty:
                        assistant_reply = (
                            "I couldn't find any clear matches for that question in this dataset. "
                            "You might try rephrasing it or asking about a specific symptom or condition."
                        )
                    else:
                        intro = (
                            f"Here's what I was able to understand from similar cases in this dataset about "
                            f"**{user_query}**. I’ll give you a warm, easy-to-follow summary of what clinicians "
                            "noted in these situations.\n\n"
                            "**Just a reminder:** this is only for learning and exploration, not medical advice.\n\n"
                        )

                        case_summaries = []
                        for i, (_, row) in enumerate(results.iterrows(), start=1):
                            sample_name = row.get("sample_name", f"Case {i}")
                            specialty = row.get("medical_specialty", "Unknown specialty")

                            description = row.get("description", "")
                            if isinstance(description, str) and description.strip():
                                desc_text = description.strip()
                            else:
                                desc_text = "No short description was provided in this note."

                            snippet = ""
                            if isinstance(row.get("clean_text"), str):
                                snippet = row["clean_text"].strip().replace("\n", " ")
                                if len(snippet) > 260:
                                    snippet = snippet[:260].rsplit(" ", 1)[0] + "..."

                            block = (
                                f"### 🩺 Case {i}: {sample_name}\n"
                                f"- **Specialty:** {specialty}\n"
                                f"- **What this note focuses on:** {desc_text}\n"
                            )
                            if snippet:
                                block += f"- **How the clinician described it:** “{snippet}”\n"

                            case_summaries.append(block)

                        outro = (
                            "\nI hope this gives you a clearer, more human understanding of how similar cases "
                            "were described. If you're ever unsure or dealing with real symptoms, it's always "
                            "best to talk with a licensed healthcare professional."
                        )

                        assistant_reply = intro + "\n\n".join(case_summaries) + "\n\n" + outro

                    st.markdown(assistant_reply)

            # Save assistant reply in history
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    # Clear chat
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.experimental_rerun()
