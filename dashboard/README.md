# 🧠 Projet 8 — Dashboard de Scoring Crédit & Veille Technique (CLIP)

Projet réalisé dans le cadre du **parcours Data Scientist – OpenClassrooms**.

Il comprend :
- une **API de scoring crédit** déployée sur Azure (FastAPI),
- un **dashboard Streamlit interactif** pour l’aide à la décision,
- une **veille technique** sur le modèle multimodal **CLIP (OpenAI)**.

---

## 🚀 Accès aux applications (Cloud)

### 🔵 Dashboard Streamlit
👉 https://p8-dashboard-adri2209.streamlit.app/

Dashboard interactif destiné :
- aux **conseillers bancaires**,
- au **COMEX** (vue synthétique direction).

Aucune installation locale requise.

### 🟣 API de scoring (Azure – FastAPI)
👉 https://implementer-un-modele-de-scoring-b6fwe6eegaamhkdh.francecentral-01.azurewebsites.net/

Endpoints principaux :
- `GET /client_info/{id}`
- `POST /prediction`
- `GET /local_feature_importance/{id}`
- `GET /global_feature_importance`

---

## 📊 Fonctionnalités clés

### 🧮 Scoring Crédit
- Probabilité de défaut
- Jauge métier avec seuil
- Score de confiance de la décision

### 🧠 Explicabilité
- SHAP local & global
- Interprétation automatique en langage naturel
- Variables explicatives principales

### 👥 Analyse client
- Comparaison à la population globale
- Détection d’atypies
- Clients similaires
- Visualisations avancées (scatter, densité, radar)

### 👔 Mode COMEX
- Synthèse exécutive
- KPIs métier
- Lecture stratégique du risque

---

## 🧪 Veille Technique — CLIP (OpenAI)

Étude comparative entre **CLIP** et un **CNN classique (VGG16)**.

### 📈 Résultats principaux

| Modèle | Précision test |
|------|---------------|
| VGG16 | 74,67 % |
| **CLIP** | **77,33 %** |

👉 **CLIP surpasse la baseline** grâce à sa compréhension multimodale,  
sans data augmentation ni réentraînement.

---

## 📂 Structure simplifiée du dépôt


```text
.
├── dashboard/
│   ├── Tint_Adriana_1_dashboard_112025.py
│   └── requirements.txt
│
├── api.py
├── Tint_Adriana_2_notebook_veille_112025.ipynb
├── Tint_Adriana_3_note_methodologique_112025.pdf
└── README.md

## 🛠️ Technologies utilisées

- **Python**
- **Streamlit**
- **FastAPI**
- **Plotly**
- **scikit-learn**
- **SHAP**
- **Pandas / NumPy**
- **Azure App Services**
- **CLIP (OpenAI)**

---

## 🙋‍♀️ Auteur

**Tint Adriana**  
Data Scientist — OpenClassrooms  
📅 2025
