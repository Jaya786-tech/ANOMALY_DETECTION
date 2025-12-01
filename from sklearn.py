from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def get_baseline_models():
    """
    Returns a dict of ML/DL baseline models.
    """
    return {
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "iso": IsolationForest(contamination=0.05, random_state=42),
        "dnn": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42),
    }
