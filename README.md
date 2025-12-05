# UTC Course Mapper

Outil de matching sémantique pour la Direction des Relations Internationales (DRI) de l'UTC.

## Objectif

Lorsqu'un étudiant part en échange à l'étranger, il s'inscrit à des cours dans l'université partenaire. Ce projet permet de :

- **Identifier** les UV UTC correspondantes aux cours suivis à l'étranger
- **Suggérer** les correspondances les plus proches si aucun match exact n'existe
- **Estimer** les équivalences ECTS

## Fonctionnement

1. Extraction et indexation de toutes les UV du catalogue UTC
2. Analyse sémantique des cours étrangers
3. Recherche des UV les plus similaires par correspondance vectorielle
4. Proposition des meilleures correspondances avec score de confiance
