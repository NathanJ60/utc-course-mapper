import os
import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient

from config import QDRANT_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL


@st.cache_resource
def get_clients():
    """Initialise les clients OpenAI et Qdrant."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY non définie")
        st.stop()
    return OpenAI(api_key=api_key), QdrantClient(path=str(QDRANT_DB_PATH))


def get_embedding(text: str, client: OpenAI) -> list[float]:
    """Génère l'embedding pour un texte."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def search_uv(query: str, openai_client: OpenAI, qdrant_client: QdrantClient, top_k: int = 5):
    """Recherche les UV similaires."""
    embedding = get_embedding(query, openai_client)
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=top_k
    ).points
    return results


# Interface Streamlit
st.set_page_config(page_title="UTC Course Mapper", page_icon="🎓")
st.title("🎓 UTC Course Mapper")
st.markdown("Trouve les UV UTC correspondant à un cours étranger")

st.divider()

# Formulaire
col1, col2 = st.columns([3, 1])

with col1:
    nom = st.text_input("Nom du cours", placeholder="Ex: Data Structure")

with col2:
    credits = st.number_input("Crédits", min_value=1, max_value=30, value=6)

description = st.text_area("Description (optionnel)", placeholder="Ex: Introduction to algorithms and data structures...")

top_k = st.slider("Nombre de résultats", min_value=1, max_value=10, value=5)

# Recherche
if st.button("🔍 Rechercher", type="primary", use_container_width=True):
    if not nom:
        st.warning("Entre un nom de cours")
    else:
        openai_client, qdrant_client = get_clients()

        query = nom
        if description:
            query += " " + description

        with st.spinner("Recherche en cours..."):
            results = search_uv(query, openai_client, qdrant_client, top_k)

        st.divider()
        st.subheader(f"Top {len(results)} UV correspondantes")

        for i, r in enumerate(results, 1):
            score_pct = int(r.score * 100)

            with st.container():
                col1, col2, col3 = st.columns([1, 4, 1])

                with col1:
                    st.metric("Score", f"{score_pct}%")

                with col2:
                    st.markdown(f"### [{r.payload['code']}] {r.payload['nom']}")
                    st.caption(f"Type: {r.payload['type']} | Crédits: {r.payload['credits']} | Semestre: {r.payload['semestre']}")
                    if r.payload['description']:
                        st.markdown(f"_{r.payload['description'][:200]}..._")

                with col3:
                    st.markdown(f"**{r.payload['credits']} ECTS**")

                st.divider()
