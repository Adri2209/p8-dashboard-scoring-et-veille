import streamlit as st # type: ignore
import requests # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import base64
import io

# Mise à jour mineure pour relancer le déploiement

# URL de ton API Flask
API_URL = "https://implementer-un-modele-de-scoring-b6fwe6eegaamhkdh.francecentral-01.azurewebsites.net/"

st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")
st.title("📊 Dashboard de Scoring Crédit")

# --- Fonction utilitaire pour récupérer les données de l'API ---
def get_api_json(endpoint, method="GET", payload=None):
    try:
        if method == "GET":
            response = requests.get(f"{API_URL}/{endpoint}")
        elif method == "POST":
            response = requests.post(f"{API_URL}/{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API: {e}")
        return None

# --- Sélection du client ---
st.sidebar.header("🔎 Sélection du client")
client_list = list(range(1, 21))

client_choice = st.sidebar.selectbox(
    "Choisir un client ID",
    options=client_list,
    index=0,
    key="client_selectbox"
)

client_id_input = st.sidebar.number_input(
    "Ou entrer un ID manuellement (facultatif)",
    min_value=1,
    step=1,
    key="client_id_input"
)

# --- Bouton pour charger les infos ---
if st.button("Charger les infos client", key="load_client_btn"):
    # Si l'utilisateur a saisi un ID différent de celui du menu, on le prend
    if client_id_input != client_choice:
        selected_id = client_id_input
    else:
        selected_id = client_choice

    # Récupérer les infos client
    client_info = get_api_json(f"client_info/{selected_id}")
    if client_info:
        st.subheader("📌 Informations client")
        st.json(client_info)

        # Prédiction
        pred_data = get_api_json("prediction", method="POST", payload={"client_id": selected_id})
        if pred_data:
            prediction = pred_data["prediction"]
            st.subheader("🎯 Probabilité de défaut de paiement")
            st.metric(label="Score de défaut", value=f"{prediction:.2%}")

        # Local Feature Importance
        shap_local_data = get_api_json(f"local_feature_importance/{selected_id}")
        if shap_local_data:
            st.subheader("🔎 Importance locale des features (SHAP)")
            shap_df = pd.DataFrame(list(shap_local_data.items()), columns=["Feature", "Importance"])
            shap_df = shap_df.sort_values("Importance", ascending=False)
            st.bar_chart(shap_df.set_index("Feature"))

        # SHAP Summary Plot (image encodée en base64)
        shap_plot_data = get_api_json(f"shap_summary_plot/{selected_id}")
        if shap_plot_data:
            st.subheader("📈 Graphique SHAP")
            img_base64 = shap_plot_data["shap_summary_plot"]
            image = base64.b64decode(img_base64)
            st.image(io.BytesIO(image))
    else:
        st.error("❌ Client introuvable dans la base de test")

# --- Global Feature Importance ---
placeholder = st.empty()  # conteneur vide

if st.checkbox("Afficher l’importance globale des features", key="global_importance_chk"):
    global_data = get_api_json("global_feature_importance")
    if global_data:
        st.subheader("🌍 Importance globale des features")
        global_df = pd.DataFrame(list(global_data.items()), columns=["Feature", "Importance"])
        global_df = global_df.sort_values("Importance", ascending=False)
        placeholder.bar_chart(global_df.set_index("Feature"))
    else:
        st.error("Impossible de récupérer l’importance globale des features.")
else:
    placeholder.empty()  # vide le placeholder quand on décoche
