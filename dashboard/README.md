🧠 Projet 8 — Dashboard de Scoring Crédit & Veille Technique (Modèle CLIP)

Ce projet a été réalisé dans le cadre d’OpenClassrooms – Parcours Data Scientist.
Il comprend :

une API de scoring crédit déployée sur Azure (FastAPI),

un dashboard Streamlit interactif destiné aux conseillers bancaires et au COMEX,

une analyse de similarité clients, des visuels interactifs et des explications SHAP,

une veille technique approfondie autour du modèle CLIP (OpenAI) appliqué à la classification d’images multimodales,

plusieurs livrables (note méthodologique, rapport, notebook, PDF…).

📂 Structure du dépôt
📁 Projet8/
│
├── dashboard/
│   ├── Tint_Adriana_1_dashboard_112025.py   # Dashboard Streamlit
│   ├── requirements.txt
│   └── README.md (optionnel)
│
├── api.py                                   # API FastAPI de scoring
│
├── modèles / données (non versionnés)
│
├── Tint_Adriana_2_notebook_veille_112025.ipynb  # Notebook analyse CLIP
├── Tint_Adriana_3_note_methodologique_112025.pdf # Note méthodologique
└── README.md                                # Ce fichier

🚀 1. API de Scoring — FastAPI

L'API permet :

✔️ d’obtenir les informations d’un client
✔️ de calculer sa probabilité de défaut
✔️ d’exposer l’importance locale et globale des variables (SHAP)
✔️ de mettre à jour les données client
✔️ de fournir tous les éléments nécessaires au dashboard

🔧 Routes principales
Méthode	Endpoint	Description
GET	/client_info/{id}	Données brutes du client
POST	/prediction	Probabilité de défaut (modèle entraîné)
GET	/local_feature_importance/{id}	SHAP local
GET	/global_feature_importance	SHAP global
PUT	/client_info/{id}	Mise à jour des données client
🎛️ 2. Dashboard Streamlit

Accessible localement ou déployé sur le cloud, le dashboard propose :

🧮 Analyse du risque

jauge de probabilité de défaut

comparaison client / population globale

score de confiance de la décision

visualisations avancées : scatter, densité, atypie, radar…

🧠 Explication du modèle

SHAP local et global

interprétation automatique en langage naturel

top variables explicatives

👔 Mode COMEX (Direction)

synthèse exécutive

KPIs métier

robustesse de la décision

indicateur de risque stratégiques simplifiés

👥 Similarité clients

recherche des k clients les plus proches

comparaison du profil client vs groupes similaires

✏️ Simulation

modification des variables

recalcul automatique du score via l’API

🧪 3. Veille Technique : CLIP (OpenAI)

Une étude complète du modèle CLIP (Contrastive Language–Image Pretraining) a été réalisée :

🔍 Axes analysés

architecture duale Vision Transformer / Texte Transformer

apprentissage contrastif sur 400M paires image-texte

capacités zero-shot

comparaison avec un CNN classique (VGG16)

📊 Résultats du benchmark
Modèle	Précision test
VGG16	74,67 %
CLIP	77,33 %
🎯 Conclusion

CLIP surpasse la baseline grâce à sa compréhension multimodale, sans data augmentation ni réentraînement.

📥 Installation
1️⃣ Cloner le repository
git clone https://github.com/Adri2209/p8-dashboard-scoring-et-veille.git
cd p8-dashboard-scoring-et-veille

🌐 Accès directs (déploiement cloud)
🔵 Dashboard Streamlit (production)

👉 https://p8-dashboard-adri2209.streamlit.app/

Permet d’accéder au tableau de bord interactif sans installation locale.

🟣 API Azure FastAPI

👉 https://implementer-un-modele-de-scoring-b6fwe6eegaamhkdh.francecentral-01.azurewebsites.net/

Endpoints principaux :

/client_info/{id}

/prediction

/local_feature_importance/{id}

/global_feature_importance

💻 Exécution locale (optionnel)

2️⃣ Installer les dépendances
pip install -r dashboard/requirements.txt

3️⃣ Lancer le dashboard
streamlit run dashboard/Tint_Adriana_1_dashboard_112025.py

4️⃣ Lancer l’API (si local)
uvicorn api:app --reload

📑 Livrables fournis

✔️ Dashboard Streamlit fonctionnel
✔️ API de scoring opérationnelle
✔️ Note méthodologique complète
✔️ Notebook CLIP commenté
✔️ Rapport PDF
✔️ Captures et résultats d’expérimentations
✔️ Scripts reproductibles

🛠️ Technologies utilisées

Python

Streamlit

FastAPI

Plotly

scikit-learn

SHAP

Pandas / NumPy

Azure App Services

CLIP (OpenAI)

🙋‍♀️ Auteur

Tint Adriana
Data Scientist — OpenClassrooms
2025
