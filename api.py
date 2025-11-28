from flask import Flask, request, jsonify # type: ignore
import joblib # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import shap # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
from lightgbm import LGBMClassifier # type: ignore
import matplotlib # type: ignore

# Forcer matplotlib à utiliser un backend non interactif pour serveur
matplotlib.use('Agg') # type: ignore
import matplotlib.pyplot as plt # type: ignore
import base64
from io import BytesIO
import plotly.express as px # type: ignore

app = Flask(__name__)

# Charger les données 
test_data = pd.read_csv('test_mean_sample.csv')
train_data = pd.read_csv('train_mean_sample.csv')

# Ajouter la colonne client_id
train_data['client_id'] = range(1, len(train_data) + 1)
test_data['client_id'] = range(1, len(test_data) + 1)

# Séparer les features et la target dans les données d'entraînement
X_train = train_data.drop(['TARGET', 'client_id'], axis=1)
y_train = train_data['TARGET']

# Appliquer le scaler et SMOTE aux données d'entraînement
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Entraîner le modèle
model = LGBMClassifier(n_estimators=100, max_depth=2, num_leaves=31, force_col_wise=True)
model.fit(X_train_resampled, y_train_resampled)

# Créer un explainer SHAP basé sur le modèle entraîné
explainer = shap.Explainer(model, X_train_scaled)

# Calcul de la feature importance globale
shap_values_train = explainer(X_train_scaled, check_additivity=False)
global_shap_values = np.abs(shap_values_train.values).mean(axis=0)
global_shap_importance = pd.DataFrame(list(zip(X_train.columns, global_shap_values)), columns=['feature', 'importance']).sort_values(by='importance', ascending=False)

@app.route('/')
def home():
    return 'API pour prédire l\'accord d\'un prêt'

# Endpoint health check pour Render
@app.route('/health', methods=['GET'])
def health_check():
    return {"status": "ok"}, 200

@app.route('/check_client/<int:client_id>', methods=['GET'])
def check_client_id(client_id):
    if client_id in list(test_data['client_id']):
        return jsonify(True)
    else:
        return jsonify(False)

@app.route('/client_info/<int:client_id>', methods=['GET'])
def get_client_info(client_id):
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404
    return client_data.to_dict(orient='records')[0]

@app.route('/client_info/<int:client_id>', methods=['PUT'])
def update_client_info(client_id):
    global test_data  
    data = request.get_json()
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404
    test_data.loc[test_data['client_id'] == client_id, list(data.keys())] = list(data.values())
    return jsonify({"message": "Client information updated"}), 200

@app.route('/client_info', methods=['POST'])
def submit_new_client():
    global test_data
    data = request.get_json()
    new_client_id = len(test_data) + 1
    data['client_id'] = new_client_id
    test_data = pd.concat([test_data, pd.DataFrame(data, index=[0])], ignore_index=True)
    return jsonify({
        "message": "New client submitted",
        "client_id": new_client_id
    }), 201


@app.route('/prediction', methods=['POST'])
def get_prediction():
    data = request.get_json()
    client_id = data.get('client_id')
    if client_id is None:
        return jsonify({"error": "client_id is required"}), 400

    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    # Filtrer les colonnes inattendues
    info_client = client_data.drop('client_id', axis=1)
    
    # Appliquer les transformations et prédire
    info_client_scaled = scaler.transform(info_client)
    prediction = model.predict_proba(info_client_scaled)[0][1]
    
    return jsonify({"prediction": prediction})

@app.route('/shap_summary_plot/<int:client_id>', methods=['GET'])
def shap_summary_plot(client_id):
    try:
        client_data = test_data[test_data['client_id'] == client_id]
        if client_data.empty:
            return jsonify({"error": "Client not found"}), 404

        info_client = client_data.drop('client_id', axis=1)
        info_client_scaled = scaler.transform(info_client)

        shap_values = explainer(info_client_scaled, check_additivity=False)

        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values.values, info_client, plot_type="bar", max_display=10, show=False)

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return jsonify({"shap_summary_plot": img_base64})

    except Exception as e:
        import traceback
        print("Erreur SHAP:", traceback.format_exc())  # pour debug dans la console
        return jsonify({"error": str(e)}), 500

@app.route('/global_feature_importance', methods=['GET'])
def global_feature_importance():
    top_10_global_shap_importance = global_shap_importance.head(10)
    global_shap_importance_dict = top_10_global_shap_importance.set_index('feature')['importance'].to_dict()
    return jsonify(global_shap_importance_dict)

@app.route('/local_feature_importance/<int:client_id>', methods=['GET'])
def local_feature_importance(client_id):
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    info_client = client_data.drop('client_id', axis=1)
    info_client_scaled = scaler.transform(info_client)

    shap_values = explainer(info_client_scaled, check_additivity=False)
    local_shap_values = np.abs(shap_values.values[0])
    local_shap_importance = pd.DataFrame(list(zip(info_client.columns, local_shap_values)), columns=['feature', 'importance']).sort_values(by='importance', ascending=False)

    local_shap_importance_dict = local_shap_importance.set_index('feature')['importance'].to_dict()
    return jsonify(local_shap_importance_dict)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8000)) 
    app.run(host="0.0.0.0", port=port, debug=False)
