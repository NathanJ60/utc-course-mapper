import os
import json
import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from groq import Groq

from config import QDRANT_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

GROQ_MODEL = "llama-3.3-70b-versatile"


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


def analyze_with_llm(course_name: str, course_description: str, course_credits: int, matches: list[dict]) -> dict:
    """Analyse le Top-K avec Groq et recommande la meilleure UV."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return {"error": "GROQ_API_KEY non définie"}

    client = Groq(api_key=groq_key)

    uv_list = ""
    for m in matches:
        uv_list += f"""
- [{m['code']}] {m['nom']}
  Type: {m['type']} | Crédits: {m['credits']} | Score: {m['score']:.0%}
  Description: {m['description'][:200] if m['description'] else 'N/A'}
"""

    prompt = f"""Tu es expert en équivalences de cours universitaires pour l'UTC.

COURS ÉTRANGER:
- Nom: {course_name}
- Description: {course_description or 'Non fournie'}
- Crédits: {course_credits} ECTS

UV UTC CANDIDATES:
{uv_list}

Analyse ces UV et détermine laquelle correspond le mieux au cours étranger.
Si aucune ne correspond vraiment, dis-le.

Réponds en JSON:
{{"is_match": true/false, "code": "CODE" ou null, "nom": "Nom UV" ou null, "justification": "2-3 phrases max"}}"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )

    content = response.choices[0].message.content

    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
    except json.JSONDecodeError:
        pass

    return {"error": "Impossible de parser la réponse", "raw": content}


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

        # Préparer les données pour l'analyse LLM
        matches = []
        for i, r in enumerate(results, 1):
            matches.append({
                'rang': i,
                'score': r.score,
                'code': r.payload['code'],
                'nom': r.payload['nom'],
                'type': r.payload['type'],
                'credits': r.payload['credits'],
                'description': r.payload['description']
            })

        # Analyse LLM
        with st.spinner("Analyse en cours..."):
            analysis = analyze_with_llm(nom, description, credits, matches)

        st.divider()

        # Afficher la recommandation
        if "error" in analysis:
            st.warning(f"Erreur d'analyse: {analysis['error']}")
        else:
            if analysis.get('is_match'):
                st.success(f"**Recommandation: [{analysis['code']}] {analysis['nom']}**")
            else:
                st.info("Aucune correspondance trouvée")
            st.markdown(f"*{analysis.get('justification', 'N/A')}*")

        st.divider()
        st.subheader(f"Top {len(results)} UV candidates")

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
