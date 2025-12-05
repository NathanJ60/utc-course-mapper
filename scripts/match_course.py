import os
import json
from openai import OpenAI
from qdrant_client import QdrantClient
from groq import Groq

QDRANT_PATH = "data/qdrant_db"
COLLECTION_NAME = "uv_utc"
EMBEDDING_MODEL = "text-embedding-3-large"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5


def get_embedding(text: str, client: OpenAI) -> list[float]:
    """Génère l'embedding pour un texte."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def analyze_with_llm(course_name: str, course_description: str, course_credits: int, matches: list[dict]) -> dict:
    """Analyse le Top-5 avec Groq et recommande la meilleure UV."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return {"error": "GROQ_API_KEY non définie"}

    client = Groq(api_key=groq_key)

    # Construire le contexte
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

    # Extraire le JSON de la réponse (peut contenir du texte avant/après)
    try:
        # Chercher le JSON dans la réponse
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
    except json.JSONDecodeError:
        pass

    return {"error": "Impossible de parser la réponse", "raw": content}


def match_course(nom: str, description: str = "", credits: int = None) -> list[dict]:
    """Trouve les UV UTC les plus similaires à un cours étranger."""

    # Client OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY non définie")
    openai_client = OpenAI(api_key=api_key)

    # Client Qdrant
    qdrant_client = QdrantClient(path=QDRANT_PATH)

    # Créer le texte à vectoriser
    text = nom
    if description:
        text += " " + description

    print(f"Recherche pour: {text}")
    print(f"Crédits: {credits}")
    print("-" * 50)

    # Générer l'embedding
    embedding = get_embedding(text, openai_client)

    # Rechercher dans Qdrant
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=TOP_K
    ).points

    # Formater les résultats
    matches = []
    for i, result in enumerate(results, 1):
        match = {
            'rang': i,
            'score': round(result.score, 4),
            'code': result.payload['code'],
            'nom': result.payload['nom'],
            'type': result.payload['type'],
            'credits': result.payload['credits'],
            'description': result.payload['description']
        }
        matches.append(match)

        print(f"\n#{i} [{result.payload['code']}] {result.payload['nom']}")
        print(f"   Score: {result.score:.4f} | Type: {result.payload['type']} | Crédits: {result.payload['credits']}")
        if result.payload['description']:
            print(f"   {result.payload['description'][:100]}...")

    return matches


if __name__ == '__main__':
    # Test avec le cours étranger
    cours_etranger = {
        'nom': 'Databases',
        'description': 'Compulsory bachelor degree course',
        'credits': 6
    }

    matches = match_course(
        nom=cours_etranger['nom'],
        description=cours_etranger['description'],
        credits=cours_etranger['credits']
    )

    # Analyse avec Groq
    print("\n" + "=" * 50)
    print("ANALYSE LLM (Groq)")
    print("=" * 50)

    analysis = analyze_with_llm(
        course_name=cours_etranger['nom'],
        course_description=cours_etranger['description'],
        course_credits=cours_etranger['credits'],
        matches=matches
    )

    if "error" in analysis:
        print(f"Erreur: {analysis['error']}")
    else:
        if analysis.get('is_match'):
            print(f"\n✅ RECOMMANDATION: [{analysis['code']}] {analysis['nom']}")
        else:
            print("\n❌ Aucune correspondance trouvée")
        print(f"\n📝 Justification: {analysis.get('justification', 'N/A')}")
