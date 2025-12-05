import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

EMBEDDINGS_FILE = "data/uv_embeddings.json"
QDRANT_PATH = "data/qdrant_db"
COLLECTION_NAME = "uv_utc"
VECTOR_SIZE = 3072  # text-embedding-3-large


def main():
    # Charger les UV avec embeddings
    with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
        uvs = json.load(f)

    print(f"Chargé {len(uvs)} UV avec embeddings")

    # Créer le client Qdrant (local persisté)
    client = QdrantClient(path=QDRANT_PATH)

    # Supprimer la collection si elle existe
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' supprimée")

    # Créer la collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' créée")

    # Préparer les points
    points = []
    for i, uv in enumerate(uvs):
        point = PointStruct(
            id=i,
            vector=uv['embedding'],
            payload={
                'code': uv['code'],
                'nom': uv['nom'],
                'type': uv['type'],
                'credits': uv['credits'],
                'semestre': uv['semestre'],
                'description': uv['description'],
                'mots_cles': uv['mots_cles']
            }
        )
        points.append(point)

    # Insérer les points
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexé {len(points)} UV dans Qdrant")

    # Vérifier
    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection info: {info.points_count} points")


if __name__ == '__main__':
    main()
