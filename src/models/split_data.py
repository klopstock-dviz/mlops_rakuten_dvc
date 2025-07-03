import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Get the root directory of the project
root_dir = Path(__file__).resolve()
data_dir_scr = root_dir.parent.parent / "data"
data_dir_out = root_dir.parent.parent.parent / "data/processed_data/_split"


def main(subset_size_frac):
    X_train = pd.read_csv(data_dir_scr/'X_train_update.csv',dtype={"productid": int,	"imageid": int})
    y_train = pd.read_csv(data_dir_scr/'Y_train_CVw08PX.csv', dtype={"prdtypecode": int})

    dataset = pd.concat([X_train.drop(columns=['Unnamed: 0', 'description']), y_train.drop(columns=['Unnamed: 0'])], axis=1)

    dataset=dataset.head(int(len(dataset)*subset_size_frac)).reset_index(drop=True)
    print(f"df tail:\n{dataset.tail()}")


    # Preprocessing: Fill NaN values and split data
    dataset['designation'] = dataset['designation'].fillna('')
    X = dataset['designation']
    y = dataset['prdtypecode']

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


    pd.DataFrame(X_train).reset_index(drop=True).to_csv(data_dir_out/"X_train.csv")
    pd.DataFrame(X_test).reset_index(drop=True).to_csv(data_dir_out/"X_test.csv")
    pd.DataFrame(y_train).reset_index(drop=True).to_csv(data_dir_out/"y_train.csv")
    pd.DataFrame(y_test).reset_index(drop=True).to_csv(data_dir_out/"y_test.csv")


if __name__=="__main__":
    subset_size_frac=os.environ.get("train_size") or 0.25
    main(subset_size_frac)