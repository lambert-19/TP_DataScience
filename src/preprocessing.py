import pandas as pd
import string
import spacy
import re

# Chargement du modèle de langue anglaise SpaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Modèle SpaCy 'en_core_web_sm' non trouvé. Téléchargement en cours...")
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def load_data(file_path: str = "../data/inaug_speeches.csv") -> pd.DataFrame:
    """
    Charge les discours présidentiels depuis le fichier CSV avec l'encodage correct.
    """
    try:
        df = pd.read_csv(file_path, index_col=0, encoding='latin1')
        return df
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        raise


def clean_raw_text(text: str) -> str:
    """
    Nettoie en profondeur le texte brut des discours pour éliminer
    les artefacts d'encodage (ex: u+0097, \x92, etc.) avant le traitement SpaCy.
    """
    if not isinstance(text, str):
        return ""

    # 1. Remplacer les variantes d'apostrophes bizarres (\x92 ou résidus écrits) par une vraie apostrophe
    text = text.replace("\x92", "'").replace("u+0092", "'")

    # 2. Supprimer explicitement les résidus textuels de type hexadécimal/unicode (ex: u+0097)
    text = re.sub(r'u\+[0-9a-fA-F]{4}', ' ', text)

    # 3. Supprimer les caractères de contrôle non-ASCII invisibles ou parasites (\x00 à \x1f et \x7f à \x9f)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)

    # 4. Remplacer les tirets longs ou bizarres par des espaces
    text = text.replace("—", " ").replace("--", " ")

    # 5. Normaliser les espaces multiples créés par les étapes de suppression
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def inspect_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inspecte le DataFrame pour identifier les valeurs manquantes et les anomalies.
    Supprime les doublons et les lignes inutilisables.
    """
    print("\n--- Inspection des données ---")
    print(f"Nombre de lignes initiales : {len(df)}")

    # 1. Analyse des valeurs manquantes
    missing = df.isnull().sum()
    if missing.any():
        print("\nValeurs manquantes détectées :")
        print(missing[missing > 0])
        df = df.dropna(subset=['text'])
        print("=> Lignes sans contenu textuel supprimées.")
    else:
        print("\nAucune valeur manquante détectée.")

    # 2. Détection des doublons
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"\nNombre de doublons : {duplicates}")
        df = df.drop_duplicates()
        print("=> Doublons supprimés.")
    else:
        print("\nAucun doublon détecté.")

    return df


def process_text_spacy(text: str):
    """
    Pipeline NLP complet utilisant SpaCy.
    Effectue : Tokenisation, Nettoyage (minuscules, stop words, ponctuations),
    Lemmatisation, POS Tagging et NER.
    """
    if not isinstance(text, str):
        return [], [], []

    # Application du pipeline SpaCy sur le texte préalablement nettoyé de ses bruits d'encodage
    doc = nlp(text)

    tokens_cleaned = []
    pos_tags = []
    entities = []

    # 1. Tokenisation, Nettoyage, Lemmatisation et POS Tagging
    for token in doc:
        # Stockage du POS Tagging de base
        pos_tags.append((token.text, token.pos_))

        # Filtrage sémantique rigoureux
        token_text = token.text.lower().strip()
        lemma_text = token.lemma_.lower().strip()

        # On vérifie que le mot et son lemme ne contiennent aucun résidu parasite
        if (
                not token.is_stop
                and token_text not in string.punctuation
                and lemma_text not in string.punctuation
                and len(token_text) > 1
                and not token_text.startswith('u+')
                and not token_text.startswith('\\')
        ):
            # Ajout du lemme propre
            tokens_cleaned.append(lemma_text)

    # 2. Reconnaissance d'Entités Nommées (NER)
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    return tokens_cleaned, pos_tags, entities


def nlp_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le traitement linguistique complet sur l'ensemble du DataFrame.
    """
    print("\n--- Lancement du traitement NLP (SpaCy) ---")
    print("Cette étape peut prendre quelques dizaines de secondes selon la taille du dataset...")

    # ÉTAPE CLÉ : On applique d'abord le nettoyage de texte brut sur la colonne
    df['text'] = df['text'].apply(clean_raw_text)

    # Lancement de la tokenisation/lemmatisation SpaCy
    res = df['text'].apply(process_text_spacy)

    df['tokens_cleaned'] = [r[0] for r in res]
    df['pos_tags'] = [r[1] for r in res]
    df['entities'] = [r[2] for r in res]

    # Reconstitution d'une chaîne textuelle propre pour l'étape TF-IDF
    df['text_cleaned'] = df['tokens_cleaned'].apply(lambda x: " ".join(x))

    print("✅ Traitement linguistique terminé avec succès.")
    return df


if __name__ == "__main__":
    path = "../data/inaug_speeches.csv"
    try:
        # 1. Acquisition et Inspection
        data = load_data(path)
        data = inspect_and_clean(data)

        # 2. Pipeline de Prétraitement linguistique complet
        data = nlp_pipeline(data)

        # Aperçu des nouvelles colonnes pour validation
        print("\n🚀 Aperçu du DataFrame enrichi et nettoyé :")
        print(data[['Name', 'tokens_cleaned', 'entities']].head(2))

    except FileNotFoundError:
        print(f"Erreur : Fichier non trouvé à '{path}'. Vérifiez votre répertoire de travail.")