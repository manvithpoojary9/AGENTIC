# ============================================================
# SEMANTIC SEARCH USING STREAMLIT + SENTENCE TRANSFORMERS + FAISS
# ============================================================

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os


# ============================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Semantic Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Semantic Search")
st.write("Search information from data.txt using AI semantic search.")


# ============================================================
# 2. FIND data.txt
# ============================================================

file_path = os.path.join(
    os.path.expanduser("~"),
    "OneDrive",
    "Desktop",
    "data.txt"
)


# ============================================================
# 3. CHECK WHETHER data.txt EXISTS
# ============================================================

if not os.path.exists(file_path):

    st.error("❌ data.txt was not found!")

    st.write("Expected location:")
    st.code(file_path)

    st.stop()


# ============================================================
# 4. LOAD data.txt
# ============================================================

try:

    with open(file_path, "r", encoding="utf-8") as file:
        documents = file.readlines()

except Exception as e:

    st.error(f"Error reading data.txt: {e}")
    st.stop()


# Remove empty lines
documents = [
    doc.strip()
    for doc in documents
    if doc.strip()
]


# Display dataset information
st.success("✅ data.txt loaded successfully!")

st.write(
    f"📄 Total text chunks: {len(documents)}"
)


# ============================================================
# 5. LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


with st.spinner("⏳ Loading AI embedding model..."):

    model = load_model()


st.success("✅ Embedding model loaded successfully!")


# ============================================================
# 6. GENERATE DOCUMENT EMBEDDINGS
# ============================================================

@st.cache_data
def generate_embeddings(documents):

    embeddings = model.encode(
        documents,
        convert_to_numpy=True
    )

    return embeddings


with st.spinner("⏳ Generating document embeddings..."):

    doc_embeddings = generate_embeddings(documents)


st.success("✅ Document embeddings generated successfully!")

st.write(
    f"📊 Embedding shape: {doc_embeddings.shape}"
)


# ============================================================
# 7. CREATE FAISS VECTOR DATABASE
# ============================================================

dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(doc_embeddings)


st.success("✅ FAISS vector database created successfully!")

st.write(
    f"📦 Total vectors stored: {index.ntotal}"
)


# ============================================================
# 8. SEARCH SECTION
# ============================================================

st.header("🔎 Search Your Data")

query = st.text_input(
    "Enter your search query:",
    placeholder="Example: What is artificial intelligence?"
)


# ============================================================
# 9. PERFORM SEMANTIC SEARCH
# ============================================================

if query:

    # Convert query into embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )


    # Number of results
    top_k = min(3, len(documents))


    # Search FAISS
    distances, indices = index.search(
        query_embedding,
        top_k
    )


    # ========================================================
    # 10. DISPLAY RESULTS
    # ========================================================

    st.subheader("📌 Semantic Search Results")


    for rank, idx in enumerate(indices[0]):

        st.markdown(
            f"### 🏆 Rank {rank + 1}"
        )

        st.info(
            documents[idx]
        )

        st.write(
            f"📏 Distance: {distances[0][rank]:.4f}"
        )

        st.divider()


# ============================================================
# 11. IF NO QUERY
# ============================================================

else:

    st.info(
        "👆 Enter a question above to search your data."
    )


# ============================================================
# 12. SIDEBAR INFORMATION
# ============================================================

with st.sidebar:

    st.header("ℹ️ About")

    st.write(
        """
        This application uses:

        🔹 Sentence Transformers
        🔹 all-MiniLM-L6-v2
        🔹 FAISS
        🔹 Semantic Search
        🔹 Streamlit

        The application converts both
        documents and your query into
        numerical vectors and finds the
        most semantically similar text.
        """
    )