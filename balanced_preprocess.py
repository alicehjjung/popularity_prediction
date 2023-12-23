import pandas as pd
from sklearn.utils import resample

df = pd.read_csv('')


class_frequencies = df['popularity_level'].value_counts()

min_frequency = class_frequencies.min()

balanced_data = pd.DataFrame()
for level in class_frequencies.index:
    class_data = df[df['popularity_level'] == level]
    sampled_data = resample(class_data, replace=True, n_samples=min_frequency, random_state=42)
    balanced_data = pd.concat([balanced_data, sampled_data])

print("original :")
print(class_frequencies)
print("\nsampling:")
print(balanced_data['popularity_level'].value_counts())

print(balanced_data.head())
new = balanced_data.sample(frac=1, random_state=42)
new.to_csv("balanced.csv",index=False)