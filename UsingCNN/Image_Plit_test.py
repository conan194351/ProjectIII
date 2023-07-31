from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.model_selection import cross_validate

genre_list = [
    "dvh",
    "hd"
    "hienho",
    "mytam",
    "trungquan",
    "vu"]
col_list = ["file_id", "id"]


train_ratio = 0.8   # Train and Valid
valid_ratio = 0.1    # Testing


# For all classes
data_full = pd.read_csv("IMAGE_trainvalid_data.csv", usecols=col_list)

X_data_full = data_full["file_id"]
Y_data_full = data_full["id"]

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=9, test_size=1/9, random_state=0)
sss.get_n_splits(X_data_full, Y_data_full)


i=1
for train_index, test_index in sss.split(X_data_full, Y_data_full):
    # print("TRAIN:", train_index, "TEST:",test_index)
    X_train, X_test = X_data_full[train_index], X_data_full[test_index]
    y_train, y_test = Y_data_full[train_index], Y_data_full[test_index]

    # Save validation and training in CSV format
    df = pd.DataFrame({"file_id" : X_test, "id" : y_test})
    df.to_csv("IMAGE_valid_data_CRV"+str(i)+".csv", index=False)
    df = pd.DataFrame({"file_id" : X_train, "id" : y_train})
    df.to_csv("IMAGE_train_data_CRV"+str(i)+".csv", index=False)
    i=i+1