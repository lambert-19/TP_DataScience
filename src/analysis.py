import pandas as pd
# On importe directement les fonctions de ton fichier preprocessing.py
from preprocessing import load_data, inspect_and_clean, process_speeches

# On importe les outils pour l'analyse IA
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer


# --- TES FONCTIONS D'ANALYSE ---

def compute_global_occurrences(df: pd.DataFrame, top_n: int = 20) -> list:
    # Utilisation de la colonne cleaned_text transformée en liste de mots
    all_words = [word for text in df['cleaned_text'] for word in str(text).split()]
    return Counter(all_words).most_common(top_n)


def compute_occurrences_per_president(df: pd.DataFrame, top_n: int = 10) -> dict:
    president_counts = {}
    grouped = df.groupby('Name')
    for name, group in grouped:
        president_words = [word for text in group['cleaned_text'] for word in str(text).split()]
        president_counts[name] = Counter(president_words).most_common(top_n)
    return president_counts


def analyze_sentiments(df: pd.DataFrame) -> pd.DataFrame:
    df['polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['subjectivity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    return df


def compute_tfidf(df: pd.DataFrame, top_n_keywords: int = 10) -> tuple:
    vectorizer = TfidfVectorizer(min_df=1)
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
    feature_names = vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df.index)

    top_tfidf_keywords = []
    for idx, row in tfidf_df.iterrows():
        top_indices = row.argsort()[::-1][:top_n_keywords]
        top_words = [feature_names[i] for i in top_indices]
        top_tfidf_keywords.append(top_words)

    df['top_tfidf_keywords'] = top_tfidf_keywords
    return tfidf_df, df


# --- LE PIPELINE GLOBAL ---
def run_full_analysis_pipeline(file_path: str) -> pd.DataFrame:
    """
    Cette fonction orchestre tout le projet :
    Elle charge, nettoie, prétraite (via preprocessing.py) puis analyse les données.
    """
    # 1. Récupération et prétraitement (Étape 2 du projet)
    df_raw = load_data(file_path)
    df_cleaned = inspect_and_clean(df_raw)
    df_nlp = process_speeches(df_cleaned)  # C'est ici qu'on applique la tokenisation/lemmatisation/POS/NER

    # 2. Exécution des analyses IA (Étape 3 du projet)
    df_analysis = analyze_sentiments(df_nlp)
    tfidf_matrix, df_final = compute_tfidf(df_analysis)

    print("\n✅ Tout le pipeline (Prétraitement + Analyse) a été exécuté avec succès !")
    return df_final


if __name__ == "__main__":
    # Point d'entrée pour tester le script
    PATH_AU_DATASET = "data/inaug_speeches.csv"

    df_resultat = run_full_analysis_pipeline(PATH_AU_DATASET)

    # Vérification rapide du résultat final
    print(df_resultat[['Name', 'polarity', 'top_tfidf_keywords']].head())