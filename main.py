import kagglehub
import pandas as pd
from os.path import join

path = kagglehub.dataset_download("thedevastator/higher-education-predictors-of-student-retention")
data = pd.read_csv(join(path, 'dataset.csv'))
print(data)