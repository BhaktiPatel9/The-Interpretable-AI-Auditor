import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def run_bias_audit():
    """
    Investigating the 'Root Cause' of AI diagnostic gaps.
    Replicating graduate research on clinical dataset anomalies.
    """
    # Load synthetic clinical data mimicking my research on 200 patient records
    # Goal: Ensure model integrity and identify potential algorithmic bias
    try:
        data = pd.read_csv('clinical_data.csv') 
    except FileNotFoundError:
        print("Please include a 'clinical_data.csv' file to run the audit.")
        return

    X = data.drop('target_diagnosis', axis=1)
    y = data['target_diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Building a model architecture similar to my demand forecasting projects
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- INTERPRETABILITY STEP ---
    # Using SHAP to make the model 'steerable' and 'transparent'
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print("✅ Audit Complete. Visualizing feature importance to ensure safety...")
    
    # Generate a Summary Plot to detect if protected attributes drive the model
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.title("AI Auditor: Feature Importance & Bias Detection")
    plt.tight_layout()
    plt.savefig('audit_summary.png')

if __name__ == "__main__":
    run_bias_audit()
