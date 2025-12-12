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
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index file not found at {INDEX_PATH}")
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found at {EMB_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    with open(INDEX_PATH, "rb") as f:
        saved = pickle.load(f)

    nn = saved["nn"]
    model_name = saved["model_name"]

    df = pd.read_csv(DATA_PATH)
    embeddings = np.load(EMB_PATH)

    # ✅ handle mismatch safely (don’t crash)
    if len(df) != embeddings.shape[0]:
        st.warning(
            f"Row mismatch: df has {len(df)} rows but embeddings has {embeddings.shape[0]} rows. "
            "Auto-trimming df to match embeddings."
        )
        df = df.iloc[: embeddings.shape[0]].reset_index(drop=True)

    model = SentenceTransformer(model_name)

    # ✅ ALWAYS return (no matter what)
    return nn, model, embeddings, df, model_name


    # sanity check
    if len(df) != embeddings.shape[0]:
       st.warning(
        f"Row mismatch: df has {len(df)} rows but embeddings has {embeddings.shape[0]} rows. "
        "Auto-trimming df to match embeddings."
    )
    df = df.iloc[:embeddings.shape[0]].reset_index(drop=True)



def semantic_search(query: str, nn, model, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    NearestNeighbors-based semantic search.
    If nn was built with cosine distance:
      similarity ≈ 1 - distance
    """
    q_emb = model.encode([query], normalize_embeddings=True)
    distances, indices = nn.kneighbors(q_emb, n_neighbors=top_k)

    distances = distances[0]
    indices = indices[0]

    results = df.iloc[indices].copy()
    results["distance"] = distances
    results["similarity"] = 1.0 - results["distance"]
    results = results.sort_values("similarity", ascending=False).reset_index(drop=True)
    return results


# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Medical NLP Assistant",
    page_icon="🩺",
    layout="wide",
)

# ------------------- GLOBAL STYLES (NO CHAT BUBBLES) -------------------
st.markdown(
    """
    <style>
    /* Make content area a bit narrower and centered */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Headings */
    h1, h2, h3 {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        letter-spacing: 0.02em;
    }

    /* Small pill-style labels */
    .tag-pill {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.25rem;
        background: rgba(148, 163, 184, 0.15);
        border: 1px solid rgba(148, 163, 184, 0.35);
    }

    /* Result card */
    .result-card {
        padding: 0.75rem 0.5rem 0.25rem 0.5rem;
    }

    /* Footer */
    .footer {
        font-size: 0.8rem;
        color: #9ca3af;
        padding-top: 1rem;
        border-top: 1px solid rgba(148, 163, 184, 0.35);
        margin-top: 2rem;
        text-align: right;
    }

    .footer a {
        color: #93c5fd;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- HEADER -------------------
col_icon, col_title = st.columns([1, 6])
with col_icon:
    st.markdown("### 🩺")
with col_title:
    st.markdown("## Medical NLP Assistant")
    st.markdown(
        """
        Explore a medical transcription dataset using **semantic search** and a **chat-style assistant**.  
        All answers are generated from the dataset and are **for educational/demo purposes only, not medical advice.**
        """
    )

st.markdown(
    """
    <div>
        <span class="tag-pill">Semantic Search</span>
        <span class="tag-pill">Clinical Notes</span>
        <span class="tag-pill">Sentence Transformers</span>
        <span class="tag-pill">Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ------------------- LOAD RESOURCES -------------------
try:
    nn, model, embeddings, df, model_name = load_index_and_model()
except Exception as e:
    st.error("Failed to load app resources.")
    st.exception(e)
    st.stop()

# Tabs for modes
search_tab, chat_tab = st.tabs(["🔍 Search Mode", "💬 Chat Mode"])


# ------------------- 🔍 SEARCH MODE -------------------
with search_tab:
    left_col, right_col = st.columns([1, 2])

    with right_col:
        st.subheader("Semantic Search")

        st.write(
            """
Type a symptom, diagnosis, or clinical description, for example:  
`"fever and chills"`, `"abdominal pain"`, `"shortness of breath"`, `"postoperative chest pain"`.
            """
        )

        example_queries = [
            "fever and chills",
            "abdominal pain",
            "shortness of breath",
            "diabetes follow up",
            "postoperative chest pain",
        ]

        query_search = st.text_input(
            "Enter a query:",
            placeholder="e.g. abdominal pain after surgery",
            key="search_input",
        )

        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            run_search = st.button("Run Search", type="primary")
        with col_btn2:
            example = st.selectbox(
                "Or pick an example:",
                ["(none)"] + example_queries,
                index=0,
                key="search_example",
            )
            if example != "(none)" and not query_search:
                query_search = example
                st.session_state["search_input"] = example

    with left_col:
        st.sidebar.header("Search Settings")
        top_k_search = st.sidebar.slider(
            "Number of results (search mode)", min_value=3, max_value=15, value=5, step=1
        )

        st.sidebar.markdown("### Example queries")
        st.sidebar.write("Use the dropdown in the main area to quickly try one.")

    if run_search:
        if not query_search.strip():
            st.warning("Please enter a query first.")
        else:
            with st.spinner("Running semantic search..."):
                results = semantic_search(query_search, nn, model, df, top_k=top_k_search)

            st.markdown(f"### 🔍 Top {len(results)} results for: `{query_search}`")

            # Optional specialty filter
            if "medical_specialty" in results.columns:
                specialties = ["All"] + sorted(
                    results["medical_specialty"].dropna().astype(str).unique().tolist()
                )
                selected_specialty = st.selectbox(
                    "Filter by medical specialty (optional):",
                    specialties,
                    key="search_specialty",
                )

                if selected_specialty != "All":
                    results = results[results["medical_specialty"].astype(str) == str(selected_specialty)]

            if results.empty:
                st.info("No matching cases were found for that query.")
            else:
                for idx, row in results.reset_index(drop=True).iterrows():
                 header = row.get("sample_name", f"Sample {idx+1}")
                 sim = float(row.get("similarity", 0.0))

                 with st.expander(f"📄 {header}  ·  similarity score: {sim:.3f}", expanded=(idx < 3)):
                  st.markdown('<div class="result-card">', unsafe_allow_html=True)

                  if "transcription" in row and isinstance(row["transcription"], str):
                   st.markdown("#### Original Transcription")
                   st.write(row["transcription"])

        # ✅ cleaned text must be INSIDE the loop, inside the expander
                  with st.expander("Show model input (cleaned text)"):
                   if "clean_text" in row and isinstance(row["clean_text"], str):
                    st.write(row["clean_text"])

                  st.markdown("</div>", unsafe_allow_html=True)


    # 👇 cleaned text hidden by default
    with st.expander("Show model input (cleaned text)"):
        if "clean_text" in row and isinstance(row["clean_text"], str):
            st.write(row["clean_text"])

    st.markdown("</div>", unsafe_allow_html=True)



# ------------------- 💬 CHAT MODE -------------------
with chat_tab:
    st.subheader("Chat with the Dataset")

    st.caption(
        "Ask high-level questions like: "
        "`What are common causes of fever?`, "
        "`How is postoperative chest pain described?`, "
        "`What symptoms are associated with shortness of breath?`"
    )

    # Sidebar info
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

        normalized = user_query.lower().strip()
        greeting_keywords = {"hi", "hello", "hey"}

        # ---- Greeting branch ----
        if normalized in greeting_keywords:
            assistant_reply = (
                "Hi! 👋\n\n"
                "I'm your medical NLP assistant. You can ask things like:\n\n"
                "- `What are common causes of fever?`\n"
                "- `How is abdominal pain described?`\n"
                "- `What symptoms are associated with chest pain?`\n\n"
                "**Reminder:** this is for educational/demo purposes only — not medical advice."
            )
            with st.chat_message("assistant"):
                st.markdown(assistant_reply)

            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

        # ---- Semantic search answer branch ----
        else:
            with st.chat_message("assistant"):
                with st.spinner("Looking for similar cases in the dataset..."):
                    results = semantic_search(user_query, nn, model, df, top_k=5)

                    if results.empty:
                        assistant_reply = (
                            "I couldn't find any clear matches for that question in this dataset. "
                            "Try rephrasing it, or asking about a specific symptom/condition."
                        )
                    else:
                        intro = (
                            f"Here’s what I found in similar notes about **{user_query}**.\n\n"
                            "**Reminder:** this is only for learning/demo — not medical advice.\n\n"
                        )

                        case_summaries = []
                        for i, row in results.iterrows():
                            sample_name = row.get("sample_name", f"Case {i+1}")
                            specialty = row.get("medical_specialty", "Unknown specialty")
                            sim = float(row.get("similarity", 0.0))

                            description = row.get("description", "")
                            desc_text = description.strip() if isinstance(description, str) and description.strip() else \
                                "No short description was provided in this note."

                            snippet = ""
                            # Prefer transcription for snippet if available
                            if isinstance(row.get("transcription"), str) and row["transcription"].strip():
                                snippet = row["transcription"].strip().replace("\n", " ")
                            elif isinstance(row.get("clean_text"), str) and row["clean_text"].strip():
                                snippet = row["clean_text"].strip().replace("\n", " ")

                            if snippet:
                                if len(snippet) > 260:
                                    snippet = snippet[:260].rsplit(" ", 1)[0] + "..."

                            block = (
                                f"### 🩺 Case {i+1}: {sample_name}\n"
                                f"- **Specialty:** {specialty}\n"
                                f"- **Similarity:** {sim:.3f}\n"
                                f"- **What this note focuses on:** {desc_text}\n"
                            )
                            if snippet:
                                block += f"- **Snippet:** “{snippet}”\n"

                            case_summaries.append(block)

                        outro = (
                            "\nIf you want, tell me what kind of output you prefer next time: "
                            "**short bullets**, **a structured summary**, or **only the top 1–2 cases**."
                        )

                        assistant_reply = intro + "\n\n".join(case_summaries) + "\n\n" + outro

                    st.markdown(assistant_reply)

            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    # Clear chat button
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# ------------------- FOOTER -------------------
st.markdown(
    """
    <div class="footer">
        Built with Streamlit • Educational/Demo only
    </div>
    """,
    unsafe_allow_html=True,
)
