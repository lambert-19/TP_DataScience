import pandas as pd
import spacy
import re

# Chargement du modèle de langue anglais
nlp = spacy.load("en_core_web_sm")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Au cas où le modèle n'est pas installé
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def load_data(file_path: str = "../data/inaug_speeches.csv") -> pd.DataFrame:
def load_data(file_path: str = "data/inaug_speeches.csv") -> pd.DataFrame:
    """
    Charge les discours présidentiels depuis le fichier CSV avec l'encodage correct.
    """
    try:
        # Gestion du chemin relatif pour éviter les erreurs selon le dossier d'exécution
        if not os.path.exists(file_path) and os.path.exists("../" + file_path):
            file_path = "../" + file_path
        # Utilisation de encoding='latin1' pour éviter les UnicodeDecodeError
        # index_col=0 car la première colonne est un index numérique
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
        # Suppression des lignes où le discours est vide
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

def process_speeches(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """
    Exécute le pipeline NLP : nettoyage, lemmatisation, POS tagging et NER.
    Pipeline complet : nettoyage, lemmatisation (minuscules/stop words/ponctuation), POS et NER.
    """
    print("\n--- Prétraitement NLP (Nettoyage, Lemmatisation, POS, NER) ---")
    
    # Nettoyage initial : suppression des caractères spéciaux (ex: <U+0097>, )
    df[text_col] = df[text_col].apply(lambda x: re.sub(r'<U\+[0-9A-F]+>', ' ', str(x)))
    df[text_col] = df[text_col].apply(lambda x: x.replace('', '').strip())

    cleaned_texts = []
    all_pos = []
    all_ner = []

    # Utilisation de nlp.pipe pour traiter les textes efficacement par lots
    # disable=["parser"] accélère le traitement si on n'a pas besoin de l'arbre syntaxique
    # nlp.pipe est beaucoup plus rapide pour traiter de gros volumes de texte
    for doc in nlp.pipe(df[text_col], batch_size=10, disable=["parser"]):
        # 1. Nettoyage sémantique, Tokenisation et Lemmatisation
        # On filtre les stop words, la ponctuation et on ne garde que les jetons alphabétiques
        tokens = [token.lemma_.lower() for token in doc 
                  if not token.is_stop and not token.is_punct and not token.is_space and token.is_alpha]
        cleaned_texts.append(" ".join(tokens))
        
        # 2. POS Tagging (Part-of-Speech)
        all_pos.append([(token.text, token.pos_) for token in doc if not token.is_space])
        
        # 3. NER (Named Entity Recognition)
        all_ner.append([(ent.text, ent.label_) for ent in doc.ents])

    df['cleaned_text'] = cleaned_texts
    df['pos_tags'] = all_pos
    df['entities'] = all_ner
    
    print("=> Prétraitement terminé.")
    return df

def save_processed_data(df: pd.DataFrame, output_path: str = "data/processed_speeches.pickle"):
    """
    Sauvegarde le DataFrame traité. 
    Note: On utilise 'pickle' car les colonnes NER/POS contiennent des listes d'objets complexes.
    """
    try:
        df.to_pickle(output_path)
        print(f"✅ Données traitées sauvegardées dans : {output_path}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {e}")

if __name__ == "__main__":
    # Note : lancez le script depuis la racine du projet pour que ce chemin fonctionne
    path = "../data/inaug_speeches.csv" 
    path = "data/inaug_speeches.csv" 
    try:
        data = load_data(path)
        data = inspect_and_clean(data)
        data = process_speeches(data)
        
        print("\n✅ Aperçu des données (colonnes principales) :")
        print(data[['Name', 'Inaugural Address', 'Date', 'text']].head())
        print("\n✅ Aperçu des données après nettoyage (3 premières lignes) :")
        print(data[['Name', 'Date', 'cleaned_text']].head(3))
        print("\nExemple d'entités nommées extraites du premier discours :")
        print(data['entities'].iloc[0][:5])
        save_processed_data(data)

        print("\n--- Résumé ---")
        print(f"Discours traités : {len(data)}")
        print(f"Colonnes générées : {', '.join(data.columns)}")

    except FileNotFoundError:
        print(f"Erreur : Fichier non trouvé à '{path}'. Vérifiez votre répertoire de travail.")
        print(f"Erreur : Fichier non trouvé à '{path}'.")