import xgboost as xgb
from sklearn.metrics import f1_score
from itertools import product
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV

from pathlib import Path
import json

# Get the root directory of the project
root_dir = Path(__file__).resolve()
data_dir_in_1 = root_dir.parent.parent / "data/processed_data/_split"
data_dir_in_2 = root_dir.parent.parent / "data/processed_data/_vectorized"
data_dir_out = root_dir.parent


def main():
    y_train = pd.read_csv(data_dir_in_1/'y_train.csv')
    y_test = pd.read_csv(data_dir_in_1/'y_test.csv')

    with open(data_dir_in_2/ 'X_train_tfidf.pkl', 'rb') as f:
        X_train_tfidf = pickle.load(f)

    with open(data_dir_in_2/ 'X_test_tfidf.pkl', 'rb') as f:
        X_test_tfidf = pickle.load(f)


    # Définition des paramètres pour le GridSearch
    param_grid = {
        'max_depth': [3, 6],#, 9],
        'learning_rate': [0.01, 0.1, ],#0.2],
        'n_estimators': [100, 200, ],#300],
        'gamma': [0, 0.1, ],#0.2],
        'subsample': [0.6, 0.8, ],#1.0],
        'colsample_bytree': [0.6, 0.8,],# 1.0]
    }

    # Création du modèle XGBoost
    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(y_train["0"]))
    # xgb_model = xgb.XGBClassifier()

    # Configuration du GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=2,
        n_jobs=-1,
        verbose=0
    )

    # Exécution du GridSearch
    grid_search.fit(X_train_tfidf, y_train['0'])



    # Affichage des meilleurs paramètres et du meilleur score
    print("Meilleurs paramètres : ", grid_search.best_params_)
    
    # Évaluation du meilleur modèle sur l'ensemble de test
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_tfidf)

    best_f1_score=f1_score(y_true=y_test["0"], y_pred=y_pred, average="weighted")
    print("F1 score weighted XGB: ", best_f1_score)

    (data_dir_out/"models/best_params").mkdir(parents=True, exist_ok=True)
    with open(data_dir_out/ 'models/best_params/best_params.pkl', 'wb') as f:
        pickle.dump(grid_search.best_params_, f)

    (data_dir_out/"models/best_models").mkdir(parents=True, exist_ok=True)
    with open(data_dir_out/ 'models/best_models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)       

    (data_dir_out/"metrics").mkdir(parents=True, exist_ok=True)
    f1_score_path = data_dir_out / 'metrics/best_f1_score.json'
    with open(f1_score_path, 'w') as f:
        json.dump({"f1_score_weighted": best_f1_score}, f)


if __name__=="__main__":
    main()