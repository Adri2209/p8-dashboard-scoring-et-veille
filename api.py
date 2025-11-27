from flask import Flask, request, jsonify # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import shap # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
from lightgbm import LGBMClassifier # type: ignore
import matplotlib # type: ignore

matplotlib.use('Agg')
import matplotlib.pyplot as plt # type: ignore
from io import BytesIO

app = Flask(__name__)

# =============================
# CHARGEMENT DES DONNÉES
# =============================

test_data = pd.read_csv('test_mean_sample.csv')
train_data = pd.read_csv('train_mean_sample.csv')

# =============================
# AJOUT AGE_YEARS
# =============================

for df in [train_data, test_data]:
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (df["DAYS_BIRTH"].abs() / 365).round().astype("Int64")

# =============================
# AJOUT ID CLIENT
# =============================

train_data["client_id"] = range(1, len(train_data) + 1)
test_data["client_id"] = range(1, len(test_data) + 1)

# =============================
# ENTRAÎNEMENT DU MODÈLE
# =============================

X_train = train_data.drop(["TARGET", "client_id", "AGE_YEARS"], axis=1, errors="ignore")
y_train = train_data["TARGET"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

model = LGBMClassifier(n_estimators=100, max_depth=2, num_leaves=31, force_col_wise=True)
model.fit(X_train_resampled, y_train_resampled)

# =============================
# SHAP
# =============================

explainer = shap.Explainer(model, X_train_scaled)
shap_values_train = explainer(X_train_scaled, check_additivity=False)

global_shap = np.abs(shap_values_train.values).mean(axis=0)
global_shap_importance = pd.DataFrame(
    list(zip(X_train.columns, global_shap)),
    columns=["feature", "importance"]
).sort_values(by="importance", ascending=False)

# =============================
# ROUTES
# =============================

@app.route("/")
def home():
    return "API Scoring Crédit"

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/check_client/<int:client_id>", methods=["GET"])
def check_client(client_id):
    return jsonify(client_id in list(test_data["client_id"]))


# =============================
# INFOS CLIENT
# =============================

@app.route("/client_info/<int:client_id>", methods=["GET"])
def get_client_info(client_id):
    client_row = test_data[test_data["client_id"] == client_id]

    if client_row.empty:
        return jsonify({"error": "Client non trouvé"}), 404

    client = client_row.iloc[0].to_dict()

    # ✅ MASQUAGE DES ÂGES ABSURDES
    days = client.get("DAYS_BIRTH")

    if days is not None:
        age = int(abs(days) // 365)
        client["AGE_YEARS"] = age if 10 <= age <= 120 else None
    else:
        client["AGE_YEARS"] = None

    return jsonify(client)


@app.route("/client_info/<int:client_id>", methods=["PUT"])
def update_client(client_id):
    data = request.get_json()

    if client_id not in list(test_data["client_id"]):
        return jsonify({"error": "Client non trouvé"}), 404

    test_data.loc[test_data["client_id"] == client_id, data.keys()] = data.values()
    return jsonify({"message": "Client mis à jour"})


@app.route("/client_info", methods=["POST"])
def add_client():
    global test_data
    data = request.get_json()
    data["client_id"] = len(test_data) + 1
    test_data = pd.concat([test_data, pd.DataFrame([data])], ignore_index=True)
    return jsonify({"client_id": data["client_id"]}), 201


# =============================
# PRÉDICTION
# =============================

@app.route("/prediction", methods=["POST"])
def predict():
    data = request.get_json()
    client_id = data.get("client_id")

    row = test_data[test_data["client_id"] == client_id]
    if row.empty:
        return jsonify({"error": "Client non trouvé"}), 404

    X = row.drop(["client_id", "AGE_YEARS"], axis=1, errors="ignore")
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0][1]
    return jsonify({"prediction": float(proba)})


# =============================
# SHAP LOCAL
# =============================

@app.route("/local_feature_importance/<int:client_id>", methods=["GET"])
def local_importance(client_id):

    row = test_data[test_data["client_id"] == client_id]
    if row.empty:
        return jsonify({"error": "Client non trouvé"}), 404

    X = row.drop(["client_id", "AGE_YEIRS"], axis=1, errors="ignore")
    X_scaled = scaler.transform(X)

    shap_values = explainer(X_scaled, check_additivity=False)
    values = np.abs(shap_values.values[0])

    df = pd.DataFrame({
        "feature": X.columns,
        "importance": values
    }).sort_values(by="importance", ascending=False)

    return jsonify(df.set_index("feature").to_dict()["importance"])


# =============================
# SHAP IMAGE
# =============================

@app.route("/shap_summary_plot/<int:client_id>", methods=["GET"])
def shap_summary(client_id):
    row = test_data[test_data["client_id"] == client_id]
    if row.empty:
        return jsonify({"error": "Client non trouvé"}), 404

    X = row.drop(["client_id", "AGE_YEARS"], axis=1, errors="ignore")
    X_scaled = scaler.transform(X)

    shap_values = explainer(X_scaled, check_additivity=False)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values.values, X, plot_type="bar", max_display=10, show=False)

    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)

    img = base64.b64encode(buffer.read()).decode()
    plt.close()

    return jsonify({"shap_summary_plot": img})


@app.route("/global_feature_importance", methods=["GET"])
def global_importance():
    return jsonify(global_shap_importance.head(15).set_index("feature")["importance"].to_dict())


# =============================
# DÉMARRAGE
# =============================

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
