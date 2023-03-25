import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv('dataset/data.csv')

duplicates = data.duplicated()
data = data[~duplicates]

data = data.drop(data[data["capital_run_length_total"] > 3900].index)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
under_sampler = RandomUnderSampler()
new_X, new_y = under_sampler.fit_resample(X, y)
new_data = pd.concat([new_X, new_y], axis=1)

new_data.to_csv('dataset/cleaned_data.csv')