import os
from openai import OpenAI
from qdrant_client import QdrantClient

QDRANT_PATH = "data/qdrant_db"
COLLECTION_NAME = "uv_utc"
MODEL = "text-embedding-3-large"
TOP_K = 5


def get_embedding(text: str, client: OpenAI) -> list[float]:
    """Génère l'embedding pour un texte."""
    response = client.embeddings.create(model=MODEL, input=text)
    return response.data[0].embedding


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
