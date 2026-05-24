import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report


# ==============================
# 1. DATA PREPARATION
# ==============================

def load_data(path):
    df = pd.read_csv(path)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('classification_yes', axis=1)
    y = df['classification_yes']

    return X, y


# ==============================
# 2. TRAIN TEST SPLIT + SCALING
# ==============================

def preprocess(X, y, test_size=0.25):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# ==============================
# 3. FEATURE SELECTION (RFE)
# ==============================

def rfe_feature_selection(X_train, y_train, X_test, n_features=5):

    model = LogisticRegression(max_iter=1000)

    rfe = RFE(estimator=model, n_features_to_select=n_features)

    X_train_sel = rfe.fit_transform(X_train, y_train)
    X_test_sel = rfe.transform(X_test)

    return X_train_sel, X_test_sel


# ==============================
# 4. MODEL TRAINING FUNCTION
# ==============================

def train_and_evaluate(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)

    return acc, report


# ==============================
# 5. MODEL COLLECTION
# ==============================

def get_models():

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (Linear)": SVC(kernel='linear'),
        "SVM (RBF)": SVC(kernel='rbf'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0)
    }


# ==============================
# 6. MAIN PIPELINE
# ==============================

def run_pipeline():

    # Load data
    X, y = load_data("Prep.csv")

    # Split + scale
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # Feature selection
    X_train, X_test = rfe_feature_selection(X_train, y_train, X_test, n_features=5)

    # Models
    models = get_models()

    results = []

    for name, model in models.items():

        acc, report = train_and_evaluate(model, X_train, y_train, X_test, y_test)

        results.append({
            "Model": name,
            "Accuracy": acc
        })

        print("\n====================")
        print(name)
        print("Accuracy:", acc)
        print(report)

    # Final comparison table
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="Accuracy", ascending=False)

    print("\n===== FINAL MODEL COMPARISON =====")
    print(result_df)

    return result_df


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    run_pipeline()