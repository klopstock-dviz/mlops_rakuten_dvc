from sklearn.metrics import f1_score, classification_report
from itertools import product
import pickle
import pandas as pd
from pathlib import Path
import json
import time



# Get the root directory of the project
root_dir = Path(__file__).resolve()
data_dir_in_1 = root_dir.parent.parent / "data/processed_data/_split"
data_dir_in_2 = root_dir.parent.parent / "data/processed_data/_vectorized"
dir_model=root_dir.parent.parent / "models/best_models"
dir_metrics=root_dir.parent.parent / "metrics"



def main():

    def timer(func):
        # @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t1 = time.time()
            result = func(*args, **kwargs)
            tf = time.time()
            print(f"exec time: {tf - t1}")
            return result
        return wrapper

    
    y_test = pd.read_csv(data_dir_in_1/'y_test.csv')

    with open(data_dir_in_2/ 'X_test_tfidf.pkl', 'rb') as f:
        X_test_tfidf = pickle.load(f)

    with open(dir_model/ 'best_model.pkl', 'rb') as f:
        model=pickle.load(f)


    # Impression des paramètres du modèle
    params = model.get_params()
    # for param, value in params.items():
    #     print(f"{param}: {value}")
        
    # Predict on validation data
    y_pred = timer(model.predict)(X_test_tfidf)

    # Evaluate the model
    print("MLP Classifier:\n", classification_report(y_true=y_test["0"], y_pred=y_pred))

    score= f1_score(y_true=y_test["0"], y_pred=y_pred, average="weighted")
    print("F1 score weighted: ", score )
  

    (dir_metrics).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(y_pred).to_csv(dir_metrics/ 'y_pred.csv', index=False)


    f1_score_path = dir_metrics / 'best_f1_score.json'
    with open(f1_score_path, 'w') as f:
        json.dump({"f1_score_weighted": score}, f)


if __name__=="__main__":
    main()