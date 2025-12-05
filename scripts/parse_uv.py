import fitz
import re
import json

def extract_uvs(pdf_path: str) -> list[dict]:
    """Extrait les UV depuis le PDF du catalogue."""
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text() + "\n"

    doc.close()

    uvs = []

    # Pattern : Semestre suivi de TYPE/Crédits puis CODE + Nom
    # On cherche les blocs qui commencent par Automne/Printemps et contiennent un code UV
    uv_blocks = re.split(r'(?=(?:Automne|Printemps)\n)', full_text)

    for block in uv_blocks:
        if not block.strip():
            continue

        # Chercher le code UV (format: 2-4 lettres majuscules + 1-2 chiffres)
        # Doit être suivi d'un nom commençant par une majuscule (pas un chiffre ou minuscule)
        code_match = re.search(r'\n([A-Z]{2,4}\d{1,2})\s+([A-Z][^\n]+)', block)
        if not code_match:
            continue

        code = code_match.group(1)
        nom = code_match.group(2).strip()

        # Vérifier que c'est bien une UV (doit avoir "Description brève")
        if 'Description brève' not in block:
            continue

        # Nettoyer le nom (peut être sur plusieurs lignes)
        # Chercher si la ligne suivante fait partie du nom (pas de "Description")
        full_nom_match = re.search(rf'{re.escape(code)}\s+([^\n]+(?:\n(?!Description)[^\n]+)?)', block)
        if full_nom_match:
            nom = ' '.join(full_nom_match.group(1).split())

        # Extraire le semestre
        semestre = 'Automne' if block.startswith('Automne') else 'Printemps' if block.startswith('Printemps') else None

        # Extraire le type (CS, TM, TSH, SP)
        type_match = re.search(r'\n(CS|TM|TSH|SP)\n', block)
        uv_type = type_match.group(1) if type_match else None

        # Extraire les crédits (juste après le type)
        credits_match = re.search(r'Crédits\s*(\d+)', block)
        credits = int(credits_match.group(1)) if credits_match else None

        # Extraire la description
        desc_match = re.search(r'Description brève\s*:\s*(.+?)(?=Diplômant|Niveau|$)', block, re.DOTALL)
        description = None
        if desc_match:
            description = ' '.join(desc_match.group(1).split())

        # Extraire les mots clés
        mots_cles_match = re.search(r'Mots clés\s*:\s*([^\n]+(?:\n(?!Automne|Printemps|[A-Z]{2}\d)[^\n]+)*)', block)
        mots_cles = None
        if mots_cles_match:
            mots_cles = ' '.join(mots_cles_match.group(1).split())

        uv = {
            'code': code,
            'nom': nom,
            'type': uv_type,
            'credits': credits,
            'semestre': semestre,
            'description': description,
            'mots_cles': mots_cles
        }

        # Éviter les doublons
        if not any(u['code'] == code for u in uvs):
            uvs.append(uv)

    return uvs


if __name__ == '__main__':
    pdf_path = 'catalogue-uv/uv_catalogue_extracted.pdf'
    uvs = extract_uvs(pdf_path)

    print(f"Nombre d'UV extraites : {len(uvs)}\n")

    for uv in uvs[:10]:  # Afficher les 10 premières
        print(f"[{uv['code']}] {uv['nom']}")
        print(f"  Type: {uv['type']} | Crédits: {uv['credits']} | Semestre: {uv['semestre']}")
        if uv['description']:
            print(f"  Description: {uv['description'][:100]}...")
        print()

    # Sauvegarder en JSON
    with open('data/uv_parsed.json', 'w', encoding='utf-8') as f:
        json.dump(uvs, f, ensure_ascii=False, indent=2)

    print(f"Sauvegardé dans data/uv_parsed.json")
