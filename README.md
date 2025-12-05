# UTC Course Mapper

Outil de matching sémantique pour la Direction des Relations Internationales (DRI) de l'UTC.

## Objectif

Lorsqu'un étudiant part en échange à l'étranger, il s'inscrit à des cours dans l'université partenaire. Ce projet permet de :

- **Identifier** les UV UTC correspondantes aux cours suivis à l'étranger
- **Suggérer** les correspondances les plus proches si aucun match exact n'existe
- **Justifier** les recommandations avec une analyse contextuelle

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Cours étranger │────▶│  OpenAI Embeddings│────▶│  Qdrant (HNSW)  │
│  (nom + desc)   │     │  text-embedding-3 │     │  Cosine Search  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Recommandation │◀────│  Groq LLM        │◀────│  Top-5 UV       │
│  + Justification│     │  llama-3.3-70b   │     │  candidates     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Stack technique

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Parsing | PyMuPDF | Extraction des UV depuis le PDF catalogue |
| Embeddings | OpenAI `text-embedding-3-large` | Vectorisation sémantique (3072 dim) |
| Base vectorielle | Qdrant (local) | Stockage et recherche par similarité |
| Analyse | Groq `llama-3.3-70b-versatile` | Raisonnement sur le Top-5 |
| Interface | Streamlit | Application web |

## Installation

```bash
# Cloner le repo
git clone https://github.com/NathanJ60/utc-course-mapper.git
cd utc-course-mapper

# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Configuration

Définir les variables d'environnement :

```bash
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
```

## Utilisation

### Interface web (Streamlit)

```bash
streamlit run src/app.py
```

Ouvrir http://localhost:8501

### Script CLI

```bash
python scripts/match_course.py
```

## Structure du projet

```
utc-course-mapper/
├── data/
│   ├── uv_parsed.json       # UV extraites du catalogue
│   ├── uv_embeddings.json   # Embeddings OpenAI
│   └── qdrant_db/           # Base vectorielle Qdrant
├── scripts/
│   ├── parse_uv.py          # Parser PDF → JSON
│   ├── vectorize_uv.py      # Génération embeddings
│   ├── index_qdrant.py      # Indexation Qdrant
│   └── match_course.py      # Matching + analyse LLM
├── src/
│   ├── app.py               # Interface Streamlit
│   └── config.py            # Configuration centralisée
└── requirements.txt
```

## Pipeline de données

1. **Extraction** : `parse_uv.py` extrait 382 UV du PDF catalogue
2. **Vectorisation** : `vectorize_uv.py` génère les embeddings via OpenAI
3. **Indexation** : `index_qdrant.py` stocke les vecteurs dans Qdrant
4. **Matching** : Recherche par similarité cosinus (HNSW)
5. **Analyse** : LLM analyse le Top-5 et recommande la meilleure UV

## Algorithmes

- **Similarité** : Cosine distance (standard pour embeddings texte)
- **Recherche** : HNSW (Hierarchical Navigable Small World) pour ANN
- **Scoring** : Score 0-1 (1 = identique)
