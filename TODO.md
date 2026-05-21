📋 To-Do List : Dashboard Discours Présidentiels US
🛠️ Étape 1 : Initialisation & Environnement
[X] Créer le dépôt GitHub et le cloner en local.

[X] Configurer l'environnement virtuel (venv ou conda).

[X] Créer le fichier requirements.txt final (incluant streamlit) et installer les dépendances.

[X] Télécharger les ressources NLP indispensables (modèle anglais de SpaCy ou lexiques NLTK).

📊 Étape 2 : Data Ingestion & Prétraitement (4 points)
[X] Télécharger le dataset Kaggle et le placer dans un dossier data/.

[X] Charger et inspecter les données avec pandas (gestion des valeurs manquantes ou anomalies).

[X] Écrire le pipeline de nettoyage sémantique : passage en minuscules, suppression de la ponctuation et filtrage des stop words.

[X] Implémenter la tokenisation et la lemmatisation des discours.

[X] Ajouter l'extraction linguistique : exécution du POS Tagging et de la reconnaissance d'entités nommées (NER).

🧠 Étape 3 : Traitements & Analyses IA (7 points)
[ ] Calculer les occurrences : extraire les fréquences de mots clés (globaux et par président).

[ ] Calculer les scores de sentiment : appliquer TextBlob ou VADER pour obtenir la polarité et la subjectivité de chaque texte.

[ ] Calculer le TF-IDF : utiliser scikit-learn pour faire ressortir les mots les plus spécifiques à chaque mandat.

💻 Étape 4 : Développement du Dashboard Streamlit (Bonus & Visuels)
[ ] Initialiser l'application Streamlit (app.py ou main.py).

[ ] Créer la structure de l'interface : une barre latérale (Sidebar) pour sélectionner le président ou l'époque, et des onglets pour séparer les analyses.

[ ] Intégrer les visualisations requises :

Un graphique linéaire interactif pour l'évolution temporelle des sentiments.

Un nuage de mots dynamique selon le président sélectionné.

Un graphique en barres en opposition (Mirror Bar Chart) pour comparer les mots clés via le TF-IDF.

📄 Étape 5 : Rédaction du Rapport & Bonnes Pratiques
[ ] Structurer le code final : basculer les fonctions lourdes dans des modules propres (dossier src/) pour respecter les best practices.

[ ] Rédiger le rapport PDF : documenter ta démarche technique, y ajouter des captures d'écran de tes plus beaux graphiques Streamlit, et rédiger tes analyses/insights (ex: impact des crises historiques sur la tonalité).

[ ] Compléter le fichier README.md : expliquer clairement comment lancer l'application Streamlit et comment installer le projet.

[ ] Push final sur GitHub.