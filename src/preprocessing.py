import pandas as pd
import string
import spacy

# Chargement du modèle de langue anglaise SpaCy
# Assure-toi d'avoir lancé : python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Modèle SpaCy 'en_core_web_sm' non trouvé. Téléchargement en cours...")
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def load_data(file_path: str = "../data/inaug_speeches.csv") -> pd.DataFrame:
    """
    Charge les discours présidentiels depuis le fichier CSV avec l'encodage correct. [cite: 14, 15, 16]
    """
    try:
        df = pd.read_csv(file_path, index_col=0, encoding='latin1')
        return df
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        raise


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

    Retourne :
        - tokens_cleaned (list) : Liste des lemmes nettoyés
        - pos_tags (list) : Liste de tuples (token, pos_tag)
        - entities (list) : Liste de tuples (entité, label_ner)
    """
    if not isinstance(text, str):
        return [], [], []

    # Application du pipeline SpaCy sur le texte
    doc = nlp(text)

    tokens_cleaned = []
    pos_tags = []
    entities = []

    # 1. Tokenisation, Nettoyage, Lemmatisation et POS Tagging
    for token in doc:
        # Stockage du POS Tagging de base (ex: NOUN, VERB, ADJ)
        pos_tags.append((token.text, token.pos_))

        # Filtrage sémantique : passage en minuscules, retrait stop words et ponctuations
        token_text = token.text.lower().strip()
        if (
                not token.is_stop
                and token_text not in string.punctuation
                and len(token_text) > 1
        ):
            # Ajout du lemme (mot racine nettoyé)
            tokens_cleaned.append(token.lemma_.lower().strip())

    # 2. Reconnaissance d'Entités Nommées (NER)
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    return tokens_cleaned, pos_tags, entities


def nlp_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le traitement linguistique complet sur l'ensemble du DataFrame.
    Crée de nouvelles colonnes enrichies pour les analyses futures.
    """
    print("\n--- Lancement du traitement NLP (SpaCy) ---")
    print("Cette étape peut prendre quelques dizaines de secondes selon la taille du dataset...")

    # Utilisation d'un zip() avec apply pour distribuer les résultats dans 3 nouvelles colonnes
    res = df['text'].apply(process_text_spacy)

    df['tokens_cleaned'] = [r[0] for r in res]
    df['pos_tags'] = [r[1] for r in res]
    df['entities'] = [r[2] for r in res]

    # Transformation des tokens nettoyés en chaîne textuelle pour faciliter le TF-IDF plus tard
    df['text_cleaned'] = df['tokens_cleaned'].apply(lambda x: " ".join(x))

    print("✅ Traitement linguistique terminé avec succès.")
    return df


if __name__ == "__main__":
    # Ajuste le chemin selon l'endroit d'où tu lances le script
    path = "../data/inaug_speeches.csv"
    try:
        # 1. Acquisition et Inspection [cite: 14, 15, 16]
        data = load_data(path)
        data = inspect_and_clean(data)

        # 2. Pipeline de Prétraitement linguistique complet
        data = nlp_pipeline(data)

        # Aperçu des nouvelles colonnes pour valider visuellement le travail
        print("\n🚀 Aperçu du DataFrame enrichi :")
        print(data[['Name', 'tokens_cleaned', 'entities']].head(2))

    except FileNotFoundError:
        print(f"Erreur : Fichier non trouvé à '{path}'. Vérifiez votre répertoire de travail.")