import json
import os
from openai import OpenAI

MODEL = "text-embedding-3-large"
INPUT_FILE = "data/uv_parsed.json"
OUTPUT_FILE = "data/uv_embeddings.json"


def create_text_for_embedding(uv: dict) -> str:
    """Crée le texte à vectoriser pour une UV."""
    parts = [uv['nom']]

    if uv.get('description'):
        parts.append(uv['description'])

    if uv.get('mots_cles'):
        parts.append(uv['mots_cles'])

    return ' '.join(parts)


def get_embeddings(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Génère les embeddings pour une liste de textes."""
    response = client.embeddings.create(
        model=MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def main():
    # Vérifier la clé API
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Erreur: OPENAI_API_KEY non définie")
        print("Exécute: export OPENAI_API_KEY='ta-clé-api'")
        return

    client = OpenAI(api_key=api_key)

    # Charger les UV
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        uvs = json.load(f)

    print(f"Chargé {len(uvs)} UV depuis {INPUT_FILE}")

    # Préparer les textes
    texts = [create_text_for_embedding(uv) for uv in uvs]

    # Générer les embeddings par batch de 100
    print(f"Génération des embeddings avec {MODEL}...")
    all_embeddings = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
        embeddings = get_embeddings(batch, client)
        all_embeddings.extend(embeddings)

    # Ajouter les embeddings aux UV
    for uv, embedding in zip(uvs, all_embeddings):
        uv['embedding'] = embedding

    # Sauvegarder
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(uvs, f, ensure_ascii=False)

    print(f"Sauvegardé {len(uvs)} UV avec embeddings dans {OUTPUT_FILE}")
    print(f"Dimension des embeddings: {len(all_embeddings[0])}")


if __name__ == '__main__':
    main()
