# Dashboard NLP - Discours Presidentiels US

Projet de Data Science et NLP permettant d'analyser les discours d'investiture des presidents des Etats-Unis.  
L'application charge un dataset CSV, nettoie les textes, applique un pipeline NLP avec SpaCy, calcule des indicateurs d'analyse textuelle, puis affiche les resultats dans un dashboard Streamlit.

## Objectifs du projet

- Charger et inspecter un dataset de discours presidentiels americains.
- Nettoyer les textes et supprimer les artefacts d'encodage.
- Appliquer un pipeline NLP complet : tokenisation, lemmatisation, stop words, POS tagging et reconnaissance d'entites nommees.
- Calculer des frequences de mots globales et par president.
- Mesurer la polarite et la subjectivite des discours avec TextBlob.
- Extraire les mots les plus representatifs avec TF-IDF.
- Visualiser les resultats dans une interface Streamlit interactive.

## Fonctionnalites

Le dashboard contient trois onglets principaux :

1. **Evolution des sentiments**
   - Courbe de polarite des discours.
   - Courbe de subjectivite.
   - Indicateurs moyens pour le president selectionne.

2. **Nuages de mots et frequences**
   - Nuage de mots dynamique pour un president.
   - Graphique en barres des mots les plus frequents.
   - Tableau des frequences exactes.

3. **Analyse comparative TF-IDF**
   - Selection de deux presidents.
   - Comparaison visuelle des mots-cles les plus caracteristiques.
   - Graphique miroir pour comparer les termes importants.

## Structure du projet

```text
TP Data Science/
+-- data/
|   +-- inaug_speeches.csv      # Dataset des discours d'investiture
+-- src/
|   +-- __init__.py
|   +-- preprocessing.py        # Chargement, nettoyage et pipeline NLP
|   +-- analysis.py             # Sentiment, occurrences et TF-IDF
|   +-- visualization.py        # Dashboard Streamlit
+-- main.py                     # Fichier actuellement vide
+-- requirements.txt            # Dependances Python
+-- TODO.md                     # Suivi des etapes du projet
+-- README.md
```

## Prerequis

- Python 3.10 ou plus recent.
- Un environnement virtuel Python.
- Le dataset `data/inaug_speeches.csv` present dans le dossier `data/`.
- Une connexion internet lors de la premiere installation du modele SpaCy, si le modele n'est pas deja installe.

## Installation

Depuis la racine du projet :

```bash
python -m venv .venv
```

Activation de l'environnement virtuel sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installation des dependances :

```bash
pip install -r requirements.txt
```

Installation du modele SpaCy anglais :

```bash
python -m spacy download en_core_web_sm
```

> Remarque : le code tente aussi de telecharger automatiquement `en_core_web_sm` si le modele est absent.

## Lancer l'application

Le fichier Streamlit se trouve dans `src/visualization.py`.  
Comme les chemins du projet sont actuellement relatifs au dossier `src`, lance l'application avec :

```powershell
cd src
streamlit run visualization.py
```

Streamlit ouvrira ensuite l'application dans le navigateur, generalement a l'adresse :

```text
http://localhost:8501
```

## Lancer le pipeline sans dashboard

Pour executer uniquement le pipeline de preprocessing et d'analyse :

```powershell
cd src
python analysis.py
```

Le pipeline effectue les etapes suivantes :

1. Chargement du fichier CSV.
2. Inspection des valeurs manquantes et des doublons.
3. Nettoyage des artefacts d'encodage.
4. Tokenisation et lemmatisation avec SpaCy.
5. POS tagging et reconnaissance d'entites nommees.
6. Analyse de sentiment avec TextBlob.
7. Calcul TF-IDF.

## Donnees utilisees

Le fichier attendu est :

```text
data/inaug_speeches.csv
```

Colonnes principales du dataset :

- `Name` : nom du president.
- `Inaugural Address` : titre du discours.
- `Date` : date du discours.
- `text` : contenu textuel du discours.

## Modules principaux

### `src/preprocessing.py`

Ce module gere :

- le chargement du dataset avec `pandas`;
- l'inspection des donnees;
- la suppression des lignes inutilisables;
- le nettoyage des caracteres parasites;
- le traitement NLP avec SpaCy;
- la creation des colonnes :
  - `pos_tags`;
  - `entities`;
  - `cleaned_text`.

### `src/analysis.py`

Ce module contient les fonctions d'analyse :

- `compute_global_occurrences()` : mots les plus frequents dans tout le corpus.
- `compute_occurrences_per_president()` : mots les plus frequents par president.
- `analyze_sentiments()` : polarite et subjectivite avec TextBlob.
- `compute_tfidf()` : extraction des mots-cles via TF-IDF.
- `run_full_analysis_pipeline()` : orchestration complete du projet.

### `src/visualization.py`

Ce module contient l'application Streamlit :

- configuration de la page;
- chargement mis en cache avec `st.cache_data`;
- sidebar de selection du president;
- graphiques Matplotlib / Seaborn;
- nuages de mots;
- comparaison TF-IDF.

## Dependances principales

- `pandas` : manipulation des donnees.
- `numpy` : calculs numeriques.
- `spacy` : pipeline NLP.
- `scikit-learn` : TF-IDF.
- `textblob` : analyse de sentiment.
- `matplotlib` et `seaborn` : visualisations.
- `wordcloud` : generation des nuages de mots.
- `streamlit` : interface interactive.

## Points d'attention

- Le premier lancement peut etre lent, car le pipeline SpaCy et le calcul TF-IDF sont appliques sur tout le dataset.
- Streamlit met les donnees analysees en cache avec `st.cache_data`, ce qui accelere les interactions suivantes.
- Le chemin du dataset est defini dans `src/visualization.py` avec `../data/inaug_speeches.csv`; il faut donc lancer l'application depuis le dossier `src`.
- Les scores de sentiment sont calcules avec TextBlob sur des textes historiques en anglais. Ils donnent une indication generale, mais ne remplacent pas une analyse linguistique fine du contexte historique.

## Etat actuel du projet

Fonctionnel :

- chargement du dataset;
- nettoyage et preprocessing NLP;
- extraction des tokens, lemmes, POS tags et entites;
- analyse de sentiment;
- calcul TF-IDF;
- dashboard Streamlit avec trois vues principales.


## Commandes utiles

Installer les dependances :

```bash
pip install -r requirements.txt
```

Installer le modele SpaCy :

```bash
python -m spacy download en_core_web_sm
```

Lancer le dashboard :

```powershell
cd src
streamlit run visualization.py
```

Lancer le pipeline d'analyse :

```powershell
cd src
python analysis.py
```
