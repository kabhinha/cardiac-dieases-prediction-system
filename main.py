import pandas as pd
import numpy as np
import pickle

filename = 'heart_prediction.sav'
loaded_model = pickle.load(open(filename, 'rb'))
path = input("Enter the path: ")
test = pd.read_excel(path)
predictions = loaded_model.predict(test)
print(f"Out of {len(test)} records, {np.count_nonzero(predictions)} suffering from heart diases and {(len(test)- np.count_nonzero(predictions))} are all right")