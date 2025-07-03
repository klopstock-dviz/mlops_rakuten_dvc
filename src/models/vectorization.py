import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pathlib import Path

# Get the root directory of the project
root_dir = Path(__file__).resolve()
data_dir_in = root_dir.parent.parent.parent / "data/processed_data/_split"
data_dir_out = root_dir.parent.parent.parent / "data/processed_data/_vectorized"


def main():
    # load X, y splited data
    X_train= pd.read_csv(data_dir_in/"X_train.csv")
    X_test= pd.read_csv(data_dir_in/"X_test.csv")

    print(f"X_train shape:")
    print(X_train.shape)

    print(f"\nX_test shape:")
    print(X_test.shape)


    # Feature extraction using TF-IDF
    X_train_text = X_train['designation'].tolist()
    X_test_text = X_test['designation'].tolist()    
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    print(f"X_train_tfidf shape:")
    print(X_train_tfidf.shape)

    print(f"\nX_test_tfidf shape:")
    print(X_test_tfidf.shape)

    data_dir_out.mkdir(parents=True, exist_ok=True)
    with open(data_dir_out/ 'X_train_tfidf.pkl', 'wb') as f:
        pickle.dump(X_train_tfidf, f)

    with open(data_dir_out/ 'X_test_tfidf.pkl', 'wb') as f:
        pickle.dump(X_test_tfidf, f)




if __name__=="__main__":
    main()