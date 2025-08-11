from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from itertools import product
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import json
import os
import time
from threadpoolctl import threadpool_limits

# limiter le nb de threads
n_cores = os.cpu_count()
n_threads = max(1, int(n_cores * 0.5))

print(f"Used threads:", n_threads)




# Get the root directory of the project
root_dir = Path(__file__).resolve()
data_dir_in_1 = root_dir.parent.parent / "data/processed_data/_split"
data_dir_in_2 = root_dir.parent.parent / "data/processed_data/_vectorized"
dir_models_out=root_dir.parent.parent/"models/best_models"
dir_params_out=root_dir.parent.parent/"models/best_params"



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

    y_train = pd.read_csv(data_dir_in_1/'y_train.csv')
    y_test = pd.read_csv(data_dir_in_1/'y_test.csv')

    with open(data_dir_in_2/ 'X_train_tfidf.pkl', 'rb') as f:
        X_train_tfidf = pickle.load(f)

    with open(data_dir_in_2/ 'X_test_tfidf.pkl', 'rb') as f:
        X_test_tfidf = pickle.load(f)


    layers=[
        [100],
        [100, 50],
        [50, 100],
        # [50, 100, 200,]
    ]

    batch_sizes=[16, 32, 64, 128]
        
    alpha_l2=[0.0001, 0.001, 0.01, 0.1]

    grid_params=list(product(layers, batch_sizes, alpha_l2))
    print(f"Nb of params: ", len(grid_params))
    history_models=[]

    i=1
    for layer, batch_size, alpha in grid_params[:2]:
        print(f'layer: {layer}, batch_size: {batch_size}, alpha: {alpha}')
        print(f"param {i}/{len(grid_params)}")
        # Define the MLPClassifier
        mlp_model = MLPClassifier(
            hidden_layer_sizes=layer, 
            max_iter=15, random_state=42, 
            early_stopping=True,
            n_iter_no_change=5,
            verbose=1,
            alpha=alpha,
            batch_size=batch_size
        )

        # Train the model
        with threadpool_limits(limits=n_threads, user_api='blas'):            
            history = timer(mlp_model.fit)(X_train_tfidf, y_train["0"])

        
        # Predict on validation data
        mlp_preds = mlp_model.predict(X_test_tfidf)


        # Evaluate the model
        #print("MLP Classifier:\n", classification_report(y_val, mlp_preds))

        print("F1 score weighted: ", f1_score(y_true=y_test["0"], y_pred=mlp_preds, average="weighted") )

        history_models.append({
            "params": {"layer": layer, "batch_size": batch_size, 'alpha': alpha},
            "history": history,
            "model": mlp_model,
            "f1-score": f1_score(y_true=y_test["0"], y_pred=mlp_preds, average="weighted")})
        
        i+=1

    # Affichage des meilleurs paramètres et du meilleur score
    df_history_models=pd.DataFrame(history_models)
    df_history_models=df_history_models.sort_values(by="f1-score", ascending=False)
    
    best_params=df_history_models.head(1)["params"].values[0]
    best_f1_score=df_history_models.head(1)["f1-score"].values[0]
    best_model=df_history_models.head(1)["model"].values[0]

    print("Meilleurs paramètres : ", best_params)
    print("Meilleur score : ", best_f1_score)
    

    (dir_params_out).mkdir(parents=True, exist_ok=True)
    with open(dir_params_out/ 'best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    (dir_models_out).mkdir(parents=True, exist_ok=True)
    with open(dir_models_out/ 'best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)       


if __name__=="__main__":
    main()