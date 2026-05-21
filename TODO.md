# To-Do List - Dashboard Discours Presidentiels US

## Etape 1 - Initialisation et environnement

- [x] Creer le depot GitHub et le cloner en local.
- [x] Configurer l'environnement virtuel avec `venv` ou `conda`.
- [x] Creer le fichier `requirements.txt` final, incluant Streamlit.
- [x] Installer les dependances du projet.
- [x] Telecharger ou configurer les ressources NLP indispensables, notamment le modele SpaCy `en_core_web_sm`.

## Etape 2 - Data ingestion et preprocessing

- [x] Telecharger le dataset et le placer dans le dossier `data/`.
- [x] Charger les donnees avec `pandas`.
- [x] Inspecter les donnees et gerer les valeurs manquantes ou anomalies.
- [x] Supprimer les doublons si necessaire.
- [x] Nettoyer les artefacts d'encodage dans les discours.
- [x] Ecrire le pipeline de nettoyage semantique : minuscules, ponctuation, stop words.
- [x] Implementer la tokenisation des discours.
- [x] Implementer la lemmatisation des discours.
- [x] Ajouter le POS tagging.
- [x] Ajouter la reconnaissance d'entites nommees (NER).
- [x] Generer une colonne `cleaned_text` exploitable pour le TF-IDF.

## Etape 3 - Traitements et analyses IA

- [x] Calculer les occurrences globales des mots-cles.
- [x] Calculer les occurrences des mots-cles par president.
- [x] Calculer les scores de sentiment avec TextBlob.
- [x] Ajouter la polarite de chaque discours.
- [x] Ajouter la subjectivite de chaque discours.
- [x] Calculer le TF-IDF avec `scikit-learn`.
- [x] Extraire les mots les plus specifiques de chaque discours ou mandat.
- [x] Integrer les analyses dans un pipeline global.

## Etape 4 - Dashboard Streamlit et visualisations

- [x] Initialiser l'application Streamlit dans `src/visualization.py`.
- [x] Configurer la page Streamlit.
- [x] Ajouter une sidebar pour selectionner un president.
- [x] Creer une structure en onglets pour separer les analyses.
- [x] Integrer un graphique lineaire pour l'evolution temporelle des sentiments.
- [x] Afficher les metriques de polarite et subjectivite pour le president selectionne.
- [x] Integrer un nuage de mots dynamique selon le president selectionne.
- [x] Integrer un graphique en barres des frequences de mots.
- [x] Integrer un tableau des frequences.
- [x] Integrer un graphique miroir pour comparer les mots-cles via le TF-IDF.
- [x] Mettre en cache le pipeline d'analyse avec `st.cache_data`.

## Etape 5 - Documentation et bonnes pratiques

- [x] Structurer le code dans des modules propres sous `src/`.
- [x] Basculer les fonctions de preprocessing dans `src/preprocessing.py`.
- [x] Basculer les fonctions d'analyse dans `src/analysis.py`.
- [x] Basculer les visualisations et l'interface dans `src/visualization.py`.
- [x] Completer le fichier `README.md`.
- [x] Expliquer dans le README comment installer le projet.
- [x] Expliquer dans le README comment lancer l'application Streamlit.
- [x] Rediger le rapport PDF avec la demarche technique.
- [ ] Ajouter des captures d'ecran du dashboard dans le rapport.
- [x] Rediger les analyses et insights metier dans le rapport.
- [x] Push final sur GitHub.
