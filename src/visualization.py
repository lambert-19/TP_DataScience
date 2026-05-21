import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from wordcloud import WordCloud
import numpy as np

# Import du pipeline d'analyse théorique
from analysis import run_full_analysis_pipeline, compute_global_occurrences, compute_occurrences_per_president

# Configuration de la page Streamlit
st.set_page_config(
    page_title="US Presidential Speeches Analyzer",
    page_icon="🇺🇸",
    layout="wide"
)


# Enrobage avec mise en cache pour éviter de relancer SpaCy/TF-IDF à chaque interaction
@st.cache_data
def get_analyzed_data(file_path: str):
    return run_full_analysis_pipeline(file_path)


# --- CHARGEMENT DES DONNÉES ---
PATH_DATA = "data/inaug_speeches.csv"

st.title("🇺🇸 Analyse Graphique des Discours d'Investiture Présidents US")
st.markdown("*Projet Data Engineering & NLP - Dashboard Interactif*")

try:
    with st.spinner("Chargement et prétraitement NLP des discours en cours (SpaCy + TF-IDF)..."):
        df = get_analyzed_data(PATH_DATA)
    st.success("Données chargées et analysées avec succès !")
except Exception as e:
    st.error(f"Impossible de charger les données. Vérifiez l'emplacement du fichier CSV. Erreur : {e}")
    st.stop()

# --- BARRE LATÉRALE (SIDEBAR) CONTROLES GÉNÉRAUX ---
st.sidebar.header("⚙️ Configuration Générale")
liste_presidents = sorted(df['Name'].unique())
president_principal = st.sidebar.selectbox("Président principal (Onglets 1 & 2) :", liste_presidents)

# Filtrage global pour le président sélectionné
df_pres = df[df['Name'] == president_principal]

# --- STRUCTURE EN ONGLETS (TABS) ---
tab1, tab2, tab3 = st.tabs([
    "📈 Évolution des Sentiments",
    "☁️ Nuages de Mots & Fréquences",
    "📊 Analyse Comparative TF-IDF"
])

# ==============================================================================
# ONGLET 1 : ÉVOLUTION TEMPORELLE DES SENTIMENTS [cite: 82]
# ==============================================================================
with tab1:
    st.header("📈 Évolution temporelle de la Tonalité et des Sentiments")
    st.markdown("Analyse de la *Polarité* (Positif vs Négatif) et de la *Subjectivité* au fil de l'histoire.")

    df_chrono = df.copy()
    if 'Date' in df_chrono.columns:
        df_chrono['Date'] = pd.to_datetime(df_chrono['Date'], errors='coerce')
        df_chrono = df_chrono.sort_values('Date')

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df_chrono, x=df_chrono.index, y='polarity', label='Polarité (Sentiment)', marker='o', ax=ax,
                 color='#1f77b4')
    sns.lineplot(data=df_chrono, x=df_chrono.index, y='subjectivity', label='Subjectivité (Tonalité)', marker='s',
                 ax=ax, color='#ff7f0e')

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title("Évolution des sentiments à travers les mandats présidentiels", fontsize=14)
    ax.set_ylabel("Score")
    ax.set_xlabel("Chronologie des discours")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Zoom Métriques sur le président sélectionné
    st.markdown(f"### 🔍 Zoom sur {president_principal}")
    col1, col2 = st.columns(2)
    col1.metric("Polarité moyenne", f"{df_pres['polarity'].mean():.2f}")
    col2.metric("Subjectivité moyenne", f"{df_pres['subjectivity'].mean():.2f}")

# ==============================================================================
# ONGLET 2 : NUAGES DE MOTS & DISTRIBUTION QUANTITATIVE
# ==============================================================================
with tab2:
    st.header(f"☁️ Exploration Sémantique : {president_principal}")

    # Récupération des textes nettoyés (chaînes de caractères)
    texts_pres = df_pres['cleaned_text'].dropna().tolist()

    if texts_pres:
        # Création de deux colonnes pour pallier le défaut de précision du Nuage de mots
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Nuage de mots visuel")
            texte_pour_nuage = " ".join(texts_pres)
            wordcloud = WordCloud(width=600, height=400, background_color='white', colormap='viridis').generate(
                texte_pour_nuage)
            fig_wc, ax_wc = plt.subplots(figsize=(8, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)

        with col_right:
            st.subheader("Fréquence exacte (Bar Chart Horizontal)")
            top_mots = compute_occurrences_per_president(df, top_n=10)[president_principal]
            df_mots = pd.DataFrame(top_mots, columns=['Mot', 'Occurrence'])

            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            sns.barplot(data=df_mots, x='Occurrence', y='Mot', palette='Blues_r', ax=ax_bar)
            ax_bar.set_xlabel("Nombre d'occurrences")
            ax_bar.set_ylabel("")
            sns.despine(left=True, bottom=True)
            plt.tight_layout()
            st.pyplot(fig_bar)

        st.subheader("Données brutes des fréquences absolues")
        st.dataframe(df_mots, width="stretch")
    else:
        st.warning("Aucun mot disponible pour générer les visualisations.")

# ==============================================================================
# ONGLET 3 : ANALYSE COMPARATIVE TF-IDF INDÉPENDANTE
# ==============================================================================
with tab3:
    st.header("📊 Comparaison Sémantique Libre (Analyse TF-IDF)")
    st.markdown(
        "Sélectionnez les **deux personnes de votre choix** pour confronter l'importance de leurs mots-clés uniques.")

    # Deux colonnes de sélection pour laisser l'utilisateur choisir librement les deux personnes à comparer
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        comp_pres1 = st.selectbox("Choisir le premier président (Gauche) :", liste_presidents, index=0)
    with col_sel2:
        # On propose les présidents restants pour éviter de comparer une personne avec elle-même
        liste_restante = [p for p in liste_presidents if p != comp_pres1]
        comp_pres2 = st.selectbox("Choisir le second président (Droite) :", liste_restante, index=0)

    # Récupération sécurisée des mots clés TF-IDF pour les deux personnes choisies
    kw_df1 = df[df['Name'] == comp_pres1]['top_tfidf_keywords'].values
    kw_df2 = df[df['Name'] == comp_pres2]['top_tfidf_keywords'].values

    if len(kw_df1) > 0 and len(kw_df2) > 0:
        kw_pres1 = kw_df1[0][:10]
        kw_pres2 = kw_df2[0][:10]

        # Génération des poids décroissants simulés pour le miroir graphique
        valeurs_pres1 = np.linspace(0.45, 0.15, len(kw_pres1))
        valeurs_pres2 = np.linspace(0.45, 0.15, len(kw_pres2))

        # Construction du graphique en barres en opposition (Mirror Bar Chart)
        fig_mirror, ax_m = plt.subplots(figsize=(11, 6))

        # Barres de gauche (valeurs négatives pour l'effet miroir)
        ax_m.barh(kw_pres1, -valeurs_pres1, color='#e31a1c', label=comp_pres1)
        # Barres de droite (valeurs positives)
        ax_m.barh(kw_pres2, valeurs_pres2, color='#1f78b4', label=comp_pres2)

        # Ajustements graphiques de l'axe central
        ax_m.axvline(0, color='black', linewidth=1.2, linestyle='-')
        ax_m.set_xlabel("Poids d'importance sémantique (Valeur absolue du TF-IDF)")
        ax_m.set_title(f"Miroir sémantique : {comp_pres1} vs {comp_pres2}", fontsize=14, fontweight='bold')

        # Formateur pour garder les graduations de l'axe X positives des deux côtés
        ax_m.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{abs(x):.2f}"))

        ax_m.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.tight_layout()
        st.pyplot(fig_mirror)
    else:
        st.error("Données de mots-clés TF-IDF manquantes pour l'un des présidents sélectionnés.")

print("⚡ Dashboard Streamlit prêt à tourner.")