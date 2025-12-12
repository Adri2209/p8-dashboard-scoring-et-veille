# -*- coding: utf-8 -*-
import base64
import io

import pandas as pd  # type: ignore
import requests  # type: ignore
import streamlit as st  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore

from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.metrics import pairwise_distances  # type: ignore

# Tentative d'import de reportlab pour le PDF (optionnel)
try:
    from reportlab.lib.pagesizes import letter  # type: ignore
    from reportlab.pdfgen import canvas  # type: ignore

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = (
    "https://implementer-un-modele-de-scoring-b6fwe6eegaamhkdh.francecentral-01.azurewebsites.net"
).rstrip("/")
THRESHOLD = 0.5  # seuil d'acceptation

st.set_page_config(page_title="Dashboard de Scoring Crédit", layout="wide")
st.title("📊 Dashboard de Scoring Crédit")
st.caption("Outil d’aide à la décision pour l’octroi de crédit")

# Mode conseiller (vue simplifiée)
simple_mode = st.sidebar.checkbox("Mode conseiller (vue simplifiée)", value=False)

# Mode COMEX
comex_mode = st.sidebar.checkbox("👔 Mode COMEX (vue Direction)", value=False)

# ============================================================
# API HELPERS
# ============================================================


def api_get(path):
    try:
        r = requests.get(f"{API_URL}{path}")
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API (GET {path}) : {e}")
        return None


def api_post(path, payload):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API (POST {path}) : {e}")
        return None


def api_put(path, payload):
    try:
        r = requests.put(f"{API_URL}{path}", json=payload)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API (PUT {path}) : {e}")
        return None


# ============================================================
# DONNÉES LOCALES
# ============================================================


@st.cache_data
def load_train() -> pd.DataFrame:
    """
    Charge le fichier d'entraînement.
    On teste deux chemins possibles : dans le repo racine et au-dessus (../)
    pour s'adapter à la structure avec le dossier 'dashboard'.
    """
    possible_paths = [
        "../train_mean_sample.csv",  # cas où le script est dans dashboard/
        "train_mean_sample.csv",  # cas où le script est à la racine
    ]
    for path in possible_paths:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue

    st.error(
        "❌ Fichier 'train_mean_sample.csv' introuvable.\n"
        "Vérifie qu'il est bien présent à la racine du projet ou au bon endroit."
    )
    return pd.DataFrame()


train_df = load_train()
if train_df.empty:
    st.stop()

usable_columns = [c for c in train_df.columns if c not in ["TARGET", "client_id"]]

# ============================================================
# JAUGE MÉTIER
# ============================================================


def display_gauge(prob: float) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": ""},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred"},
                "steps": [
                    {"range": [0, THRESHOLD * 100], "color": "lightgreen"},
                    {"range": [THRESHOLD * 100, 100], "color": "pink"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "value": THRESHOLD * 100,
                },
            },
        )
    )

    fig.update_layout(
        title="Probabilité de défaut (%)",
        title_x=0.5,
        font=dict(size=16),
        plot_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# CLIENTS SIMILAIRES
# ============================================================


def find_similar_clients(train_df, client_row, cols, k=10):
    """
    Trouve les k clients les plus proches selon certaines colonnes.
    """
    df = train_df[cols].dropna().copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    client_scaled = scaler.transform(client_row[cols].values.reshape(1, -1))

    distances = pairwise_distances(X_scaled, client_scaled)

    df["distance"] = distances.flatten()
    return df.sort_values("distance").head(k)


# ============================================================
# SIDEBAR – SÉLECTION DU CLIENT + GESTION SESSION_STATE
# ============================================================

st.sidebar.header("🔎 Sélection du client")

client_id_choice = st.sidebar.selectbox(
    "Choisir l'identifiant client", options=train_df.index.tolist()
)

load_button = st.sidebar.button("Charger / mettre à jour le client")

# Initialisation des variables de session
if "client_loaded" not in st.session_state:
    st.session_state.client_loaded = False
if "client_id" not in st.session_state:
    st.session_state.client_id = None
if "client_info" not in st.session_state:
    st.session_state.client_info = None
if "prob" not in st.session_state:
    st.session_state.prob = None

# Quand on clique sur le bouton, on va chercher les infos client + prédiction
if load_button:
    client_info_resp = api_get(f"/client_info/{client_id_choice}")
    prediction_resp = api_post("/prediction", {"client_id": int(client_id_choice)})

    if client_info_resp and prediction_resp:
        st.session_state.client_loaded = True
        st.session_state.client_id = int(client_id_choice)
        st.session_state.client_info = client_info_resp
        st.session_state.prob = prediction_resp["prediction"]
    else:
        st.session_state.client_loaded = False

# Si aucun client chargé, on affiche juste un message et on arrête
if not st.session_state.client_loaded:
    st.info(
        "Veuillez sélectionner un client puis cliquer sur "
        "**Charger / mettre à jour le client**."
    )
    st.stop()

# À partir d’ici, on est sûr d’avoir un client et une probabilité
client_id = st.session_state.client_id
client_info = st.session_state.client_info
prob = st.session_state.prob
client_series = pd.Series(client_info)

# ============================================================
# VUE COMEX – DASHBOARD DIRECTION (si activé)
# ============================================================

if comex_mode:
    st.header("📌 Synthèse Exécutive – Direction Générale")

    # KPIs direction
    decision = "ACCORDÉ" if prob < THRESHOLD else "REFUSÉ"
    risk_label = "FAIBLE" if prob < 0.3 else "MODÉRÉ" if prob < THRESHOLD else "ÉLEVÉ"
    score_metier = int((1 - prob) * 100)

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Décision crédit", decision)
    k2.metric("Risque estimé", f"{prob:.1%}")
    k3.metric("Niveau de risque", risk_label)
    k4.metric("Score métier", f"{score_metier}/100")

    # JAUGE SIMPLIFIÉE
    st.markdown("### 🎯 Positionnement du risque")
    display_gauge(prob)

    # ANALYSE DIRECTION / COMEX
    st.markdown("### 🧠 Lecture stratégique")

    if prob < 0.3:
        st.success(
            "✅ **Client à faible risque financier**\n\n"
            "Le profil présente un excellent niveau de solvabilité.\n"
            "L’exposition financière est jugée faible."
        )
    elif prob < THRESHOLD:
        st.warning(
            "⚠ **Risque maîtrisé mais à surveiller**\n\n"
            "Le client se situe proche de la zone de vigilance.\n"
            "Une validation métier complémentaire est recommandée."
        )
    else:
        st.error(
            "❌ **Risque financier élevé**\n\n"
            "Le modèle détecte un niveau de risque incompatible\n"
            "avec la politique de crédit standard."
        )

    # INDICES DE CONFIANCE
    st.markdown("### 🔐 Indicateur de robustesse de décision")

    decision_confidence = int(abs(prob - THRESHOLD) * 200)
    decision_confidence = min(100, decision_confidence)

    st.progress(decision_confidence / 100)
    st.metric("Robustesse de la décision", f"{decision_confidence}/100")

    if decision_confidence > 70:
        st.success("Décision statistiquement très fiable.")
    elif decision_confidence > 40:
        st.warning("Décision modérément robuste.")
    else:
        st.error("Décision fragile – analyse humaine recommandée.")

    # IMPACT FINANCIER SIMPLIFIÉ
    st.markdown("### 💰 Lecture financière simplifiée")

    if prob < THRESHOLD:
        st.markdown(
            "- ✅ Faible exposition financière\n"
            "- ✅ Bonne capacité de remboursement anticipée\n"
            "- ✅ Client compatible avec un développement commercial futur"
        )
    else:
        st.markdown(
            "- ❌ Exposition financière élevée\n"
            "- ⚠ Risque de défaut significatif\n"
            "- ❌ Rentabilité incertaine du dossier"
        )

    # CONCLUSION COMEX
    st.markdown("### 🏁 Conclusion Direction")

    if prob < THRESHOLD:
        st.success(
            "📌 **Dossier conforme à la politique de risque.**\n\n"
            "Aucun blocage stratégique identifié."
        )
    else:
        st.error(
            "📌 **Dossier incompatible avec la politique groupe actuelle.**\n\n"
            "Refus recommandé sans mesure compensatoire forte."
        )

    # Vue COMEX = on s’arrête ici
    st.stop()

# ============================================================
# 1. Profil client
# ============================================================

st.subheader("👤 Informations client")

if simple_mode:
    cols_to_show = [
        col
        for col in [
            "NAME_CONTRACT_TYPE",
            "CODE_GENDER",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "NAME_FAMILY_STATUS",
        ]
        if col in client_series.index
    ]
    if cols_to_show:
        st.table(client_series[cols_to_show].to_frame("Valeur"))
    else:
        st.json(client_info)
else:
    st.json(client_info)

# ============================================================
# 2. Score & score qualité
# ============================================================

st.subheader("📊 Score de crédit")

col_score, col_quality = st.columns(2)

with col_score:
    display_gauge(prob)

with col_quality:
    score_quality = int((1 - prob) * 1000)
    st.metric("Score qualité de dossier", f"{score_quality} / 1000")

    if prob < THRESHOLD:
        st.success("✅ Décision modèle : PRÊT ACCORDÉ")
    else:
        st.error("❌ Décision modèle : PRÊT REFUSÉ")

    st.metric("Écart au seuil", f"{abs(prob - THRESHOLD):.2%}")

# SCORE DE CONFIANCE MÉTIER
confidence_score = int((1 - abs(prob - THRESHOLD) / THRESHOLD) * 100)
confidence_score = max(0, min(100, confidence_score))

st.markdown(
    f"🔒 **Niveau de confiance du modèle : {confidence_score}/100**\n\n"
    "Ce score reflète la robustesse de la décision automatique :\n"
    "- Plus le score est élevé, plus la décision est stable.\n"
    "- Un score faible indique un dossier proche de la limite."
)

if confidence_score < 40:
    st.warning("⚠ Décision fragile — une analyse humaine renforcée est recommandée.")
elif confidence_score < 70:
    st.info("ℹ Décision moyennement stable — vérifications conseillées.")
else:
    st.success("✅ Décision très fiable selon le modèle.")

# SCORE DE CONFIANCE (version seuil)
st.subheader("🎯 Score de confiance de la décision")

confidence = abs(prob - THRESHOLD)
confidence_score2 = min(100, int(confidence * 200))

st.metric("Niveau de confiance", f"{confidence_score2} / 100")

if confidence_score2 > 70:
    st.success("Décision très fiable – Peu de doute statistique.")
elif confidence_score2 > 40:
    st.warning("Décision interprétable – Analyse humaine recommandée.")
else:
    st.error("Décision fragile – Vérification manuelle fortement conseillée.")

st.caption(
    "Ce score mesure à quel point la probabilité est éloignée du seuil de décision. "
    "Plus le score est élevé, plus le modèle est confiant dans sa prédiction."
)

# ============================================================
# 3. Explication automatique (texte)
# ============================================================

st.subheader("📝 Explication automatique (langage naturel)")

if prob < THRESHOLD:
    phrase_risque = (
        f"Le risque de défaut de paiement estimé est de **{prob:.0%}**, "
        f"donc **en dessous du seuil interne de {THRESHOLD:.0%}**."
    )
else:
    phrase_risque = (
        f"Le risque de défaut de paiement estimé est de **{prob:.0%}**, "
        f"donc **au-dessus du seuil interne de {THRESHOLD:.0%}**."
    )

st.markdown(phrase_risque)

shap_local = api_get(f"/local_feature_importance/{client_id}")
top_vars = []
if shap_local:
    df_local = (
        pd.DataFrame(shap_local.items(), columns=["Variable", "Importance"])
        .sort_values("Importance", ascending=False)
    )
    top_vars = df_local["Variable"].head(3).tolist()

if top_vars:
    st.markdown(
        "Les variables qui ont le plus influencé la décision du modèle sont : "
        f"**{', '.join(top_vars)}**."
    )
else:
    st.caption("Les détails SHAP ne sont pas disponibles pour ce client.")

st.markdown(
    "Ce texte est destiné à aider le conseiller à expliquer la décision de façon "
    "compréhensible pour le client, sans jargon de data science."
)

# ============================================================
# 4. Recommandations métier
# ============================================================

st.subheader("💡 Recommandations pour la suite")

recommandations = []

if prob < 0.3:
    recommandations.append(
        "Le dossier est globalement solide. Le conseiller peut mettre en avant "
        "la bonne capacité de remboursement du client."
    )
elif prob < THRESHOLD:
    recommandations.append(
        "Le risque est modéré. Il peut être utile de vérifier certains éléments "
        "du dossier (stabilité professionnelle, charges récurrentes, etc.)."
    )
else:
    recommandations.append(
        "Le risque estimé est élevé. Il est recommandé de discuter avec le client "
        "des raisons possibles et d'envisager un montant plus faible, une durée plus longue "
        "ou d'autres garanties."
    )

if "AMT_INCOME_TOTAL" in client_series.index and "AMT_CREDIT" in client_series.index:
    revenus = client_series["AMT_INCOME_TOTAL"]
    credit = client_series["AMT_CREDIT"]
    ratio = credit / revenus if revenus else None
    if ratio and ratio > 5:
        recommandations.append(
            "Le montant du crédit est très élevé par rapport aux revenus. "
            "Proposer une réduction du montant ou une durée plus longue."
        )

if "DAYS_EMPLOYED" in client_series.index:
    days_emp = client_series["DAYS_EMPLOYED"]
    if days_emp is not None and days_emp > -365:
        recommandations.append(
            "L'ancienneté professionnelle est faible. Il peut être pertinent "
            "de demander des justificatifs supplémentaires (CDI, période d'essai, etc.)."
        )

for rec in recommandations:
    st.markdown(f"- {rec}")

if not recommandations:
    st.caption("Aucune recommandation spécifique n'a été générée pour ce dossier.")

# ============================================================
# 5. Interprétation SHAP locale & globale (si pas simple_mode)
# ============================================================

if not simple_mode:
    st.subheader("🧠 Interprétation du modèle – locale (client)")

    if shap_local:
        df_local_top = (
            pd.DataFrame(shap_local.items(), columns=["Variable", "Importance"])
            .sort_values("Importance", ascending=False)
            .head(10)
        )

        fig_local = px.bar(
            df_local_top,
            x="Variable",
            y="Importance",
            title="Top 10 variables influentes (client)",
        )
        fig_local.update_layout(title_x=0.5)
        st.plotly_chart(fig_local, use_container_width=True)
    else:
        st.info("Pas de données SHAP locales disponibles.")

    st.subheader("🌍 Interprétation globale")

    shap_global = api_get("/global_feature_importance")
    if shap_global:
        df_glob = (
            pd.DataFrame(shap_global.items(), columns=["Variable", "Importance"])
            .sort_values("Importance", ascending=False)
            .head(15)
        )
        fig_glob = px.bar(
            df_glob,
            x="Variable",
            y="Importance",
            title="Variables les plus influentes (globalement)",
        )
        fig_glob.update_layout(title_x=0.5)
        st.plotly_chart(fig_glob, use_container_width=True)
    else:
        st.info("Pas de données SHAP globales disponibles.")

# ============================================================
# 6. Comparaison population globale (UNIVARIÉE)
# ============================================================

st.subheader("📈 Comparaison avec la population globale")

default_feature = st.session_state.get("feature", usable_columns[0])

feature = st.selectbox(
    "Variable à comparer",
    usable_columns,
    index=usable_columns.index(default_feature),
    key="feature",
)

fig_hist = px.histogram(train_df, x=feature, title=f"Distribution de {feature}")
if feature in client_series.index:
    fig_hist.add_vline(
        x=client_series[feature],
        line_color="red",
        line_dash="dash",
        annotation_text="Client",
        annotation_position="top",
    )
fig_hist.update_layout(title_x=0.5)
st.plotly_chart(fig_hist, use_container_width=True)

# ANALYSE PERCENTILE CLIENT
st.subheader("📊 Position du client dans la population")

if feature in train_df.columns and feature in client_series.index:
    value = client_series[feature]
    percentile = (train_df[feature] < value).mean() * 100

    st.metric("Position percentile", f"{percentile:.1f} %")

    if percentile < 10:
        st.error("Valeur très rare dans la population (<10%)")
    elif percentile < 25:
        st.warning("Valeur atypique (faible fréquence)")
    elif percentile < 75:
        st.success("Valeur courante dans la population")
    else:
        st.info("Valeur élevée par rapport à la majorité des clients")

    st.caption(
        f"Cela signifie que {percentile:.0f} % des clients ont une valeur inférieure "
        f"à celle de ce client pour cette variable."
    )

# ============================================================
# 6.b Analyse bi-variée avancée
# ============================================================

st.subheader("🔍 Analyse bi-variée avancée (corrélation, densité & atypie)")

default_x = st.session_state.get("bivar_x", usable_columns[0])

var_x = st.selectbox(
    "Variable X",
    usable_columns,
    index=usable_columns.index(default_x),
    key="bivar_x",
)

usable_y = [v for v in usable_columns if v != var_x]
if not usable_y:
    st.warning("Pas assez de variables pour une analyse bi-variée.")
    st.stop()

default_y = st.session_state.get("bivar_y", usable_y[0])
if default_y not in usable_y:
    default_y = usable_y[0]

var_y = st.selectbox(
    "Variable Y",
    usable_y,
    index=usable_y.index(default_y),
    key="bivar_y",
)

if var_x not in client_series.index or var_y not in client_series.index:
    st.error("Variables non disponibles pour ce client.")
else:
    df_bi = train_df[[var_x, var_y, "TARGET"]].dropna().copy()
    df_bi["Risque"] = df_bi["TARGET"].map({0: "Bon payeur", 1: "Défaut"})

    x_val = client_series[var_x]
    y_val = client_series[var_y]

    corr = df_bi[var_x].corr(df_bi[var_y])
    if pd.isna(corr):
        st.warning("Corrélation non calculable (données constantes ou invalides).")
    else:
        st.metric("Corrélation (Pearson)", f"{corr:.2f}")

        fig = px.scatter(
            df_bi,
            x=var_x,
            y=var_y,
            color="Risque",
            opacity=0.45,
            title=f"{var_x} vs {var_y}",
            color_discrete_map={"Bon payeur": "green", "Défaut": "red"},
        )

        fig.add_scatter(
            x=[x_val],
            y=[y_val],
            mode="markers",
            marker=dict(color="black", size=15, symbol="x"),
            name="Client",
        )

        st.plotly_chart(fig, use_container_width=True)

        def zscore(v, s):
            return (v - s.mean()) / s.std()

        z_x = zscore(x_val, df_bi[var_x])
        z_y = zscore(y_val, df_bi[var_y])

        st.markdown("### 🔎 Zone locale client")

        dx = df_bi[var_x].std()
        dy = df_bi[var_y].std()

        zone = df_bi[
            (df_bi[var_x].between(x_val - dx, x_val + dx))
            & (df_bi[var_y].between(y_val - dy, y_val + dy))
        ]

        fig_zoom = px.scatter(
            zone,
            x=var_x,
            y=var_y,
            color="Risque",
            title="Voisinage direct du client",
            color_discrete_map={"Bon payeur": "green", "Défaut": "red"},
        )

        fig_zoom.add_scatter(
            x=[x_val],
            y=[y_val],
            marker=dict(color="black", size=16, symbol="x"),
            name="Client",
        )

        st.plotly_chart(fig_zoom, use_container_width=True)

        st.markdown("### 📊 Densité de population")

        fig_density = px.density_contour(
            df_bi,
            x=var_x,
            y=var_y,
            color="Risque",
            title="Zones de densité",
        )

        fig_density.add_scatter(
            x=[x_val], y=[y_val], marker=dict(color="black", size=12), name="Client"
        )

        st.plotly_chart(fig_density, use_container_width=True)

        st.subheader("🚨 Détection d’atypie")

        for v, z in [(var_x, z_x), (var_y, z_y)]:
            if abs(z) > 3:
                st.error(f"{v} extrêmement atypique (z={z:.2f})")
            elif abs(z) > 2:
                st.warning(f"{v} atypique (z={z:.2f})")
            else:
                st.success(f"{v} dans la norme (z={z:.2f})")

        st.subheader("🧠 Lecture automatique")

        force = (
            "forte"
            if abs(corr) > 0.7
            else "modérée"
            if abs(corr) > 0.4
            else "faible"
        )
        sens = (
            "évoluent ensemble"
            if corr > 0
            else "évoluent inversement"
            if corr < 0
            else "ne sont pas liées"
        )

        st.markdown(f"- Relation **{force}**, variables qui **{sens}**.")

        if abs(z_x) > 2 or abs(z_y) > 2:
            st.warning("Positionnement atypique – examen métier recommandé.")
        else:
            st.success("Profil cohérent avec la population.")

        st.subheader("👥 Comparaison aux profils proches")

        st.metric("Profils similaires détectés", len(zone))

        if len(zone) >= 5:
            comp = pd.DataFrame(
                {
                    "Client": [x_val, y_val],
                    "Moyenne du groupe": [zone[var_x].mean(), zone[var_y].mean()],
                },
                index=[var_x, var_y],
            )

            fig_sim = px.bar(comp, barmode="group", title="Client vs proches")
            st.plotly_chart(fig_sim, use_container_width=True)

            for v in [var_x, var_y]:
                diff = abs(client_series[v] - zone[v].mean())
                std = zone[v].std()
                if diff > std:
                    st.warning(f"{v} fortement différent du groupe")
                else:
                    st.success(f"{v} cohérent")
        else:
            st.info("Pas assez de voisins pour analyse.")

# ============================================================
# 7. Clients similaires
# ============================================================

st.subheader("🧩 Clients similaires")

variables = st.multiselect(
    "Variables de similarité",
    usable_columns,
    default=[
        v
        for v in ["PAYMENT_RATE", "EXT_SOURCE_2", "INCOME_CREDIT_PERC"]
        if v in usable_columns
    ],
)

n_neighbors = st.slider("Nombre de profils similaires", 3, 30, 10)

if variables:
    client_row_train = train_df.loc[client_id]
    similar = find_similar_clients(train_df, client_row_train, variables, n_neighbors)

    st.markdown("**Liste des clients similaires (sur le train)**")
    st.dataframe(similar)

    comp_sim = pd.DataFrame(
        {
            "Client": client_row_train[variables],
            "Moyenne profils similaires": similar[variables].mean(),
        }
    )

    fig_sim2 = px.bar(
        comp_sim,
        barmode="group",
        title="Client vs profils similaires",
    )
    fig_sim2.update_layout(title_x=0.5)
    st.plotly_chart(fig_sim2, use_container_width=True)

    for var in variables:
        try:
            diff = abs(client_row_train[var] - similar[var].mean())
            std = similar[var].std()
            if pd.notna(std) and std != 0 and diff > std:
                st.warning(
                    f"⚠ **{var}** est atypique par rapport aux profils similaires."
                )
            else:
                st.success(
                    f"✅ **{var}** est cohérent avec les profils similaires."
                )
        except Exception:
            pass

# ============================================================
# 8. Radar chart (profil synthétique)
# ============================================================

st.subheader("📌 Profil synthétique (radar)")

radar_vars_default = [
    v
    for v in [
        "PAYMENT_RATE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "ANNUITY_INCOME_PERC",
        "INCOME_CREDIT_PERC",
    ]
    if v in usable_columns
]

radar_vars = st.multiselect(
    "Variables pour le radar",
    usable_columns,
    default=radar_vars_default,
)

if radar_vars:
    radar_df = train_df[radar_vars].copy()
    radar_min = radar_df.min()
    radar_max = radar_df.max()
    radar_range = radar_max - radar_min
    radar_range[radar_range == 0] = 1

    client_vals = (client_series[radar_vars] - radar_min) / radar_range
    pop_mean = (radar_df.mean() - radar_min) / radar_range

    categories = radar_vars + [radar_vars[0]]

    client_trace = list(client_vals.values) + [client_vals.values[0]]
    pop_trace = list(pop_mean.values) + [pop_mean.values[0]]

    radar_fig = go.Figure()

    radar_fig.add_trace(
        go.Scatterpolar(r=client_trace, theta=categories, fill="toself", name="Client")
    )

    radar_fig.add_trace(
        go.Scatterpolar(
            r=pop_trace, theta=categories, fill="toself", name="Moyenne population"
        )
    )

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Profil du client vs moyenne population",
    )

    st.plotly_chart(radar_fig, use_container_width=True)
else:
    st.info("Choisissez au moins une variable pour afficher le radar.")

# ============================================================
# 9. Simulation & mise à jour via API
# ============================================================

st.subheader("✏️ Simulation de modification du client")

updated = {}
with st.expander("Modifier des variables client (simulation)"):
    for k, v in client_info.items():
        if k in ["client_id", "TARGET"]:
            continue
        if isinstance(v, (int, float)):
            updated[k] = st.number_input(k, value=float(v))
        else:
            updated[k] = st.text_input(k, str(v))

    if st.button("Envoyer les modifications"):
        resp = api_put(f"/client_info/{client_id}", updated)
        if resp:
            st.success("Données mises à jour. Vous pouvez recalculer la prédiction.")

# ============================================================
# 10. Export PDF
# ============================================================

st.subheader("📄 Export du rapport (PDF)")

if not REPORTLAB_AVAILABLE:
    st.info(
        "La génération de PDF nécessite la librairie **reportlab**.\n"
        "Si vous le souhaitez, vous pourrez ajouter cette fonctionnalité plus tard "
        "en installant `reportlab` dans l'environnement serveur."
    )
else:
    if st.button("Générer le rapport PDF"):
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        text = c.beginText(40, 750)

        text.textLine("Rapport de scoring crédit")
        text.textLine(f"Client ID : {client_id}")
        text.textLine("")
        text.textLine(f"Probabilité de défaut : {prob:.2%}")
        text.textLine(f"Score qualité de dossier : {int((1 - prob) * 1000)} / 1000")
        text.textLine("")

        if top_vars:
            text.textLine("Variables principales influençant la décision :")
            for v in top_vars:
                text.textLine(f" - {v}")

        c.drawText(text)
        c.showPage()
        c.save()
        buffer.seek(0)

        st.download_button(
            "Télécharger le rapport PDF",
            data=buffer,
            file_name=f"rapport_client_{client_id}.pdf",
            mime="application/pdf",
        )
