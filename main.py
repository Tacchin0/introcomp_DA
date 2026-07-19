import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join

# dataset link
# https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data
# check it out later for infos about categorical values
# https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

# TODO
# correlation matrix
# check out either students improved or not
# some bs for economic stats idk
# improve graphics
# stuff

def standard_ax_config(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

path = kagglehub.dataset_download("thedevastator/higher-education-predictors-of-student-retention")
data = pd.read_csv(join(path, 'dataset.csv'))

# No need for pre-processing since researchers stated
# it was already been done.

data.info()

#print('Taxa de aprovação 1° semestre:',
#      data['Curricular units 1st sem (approved)'].sum()/data['Curricular units 1st sem (enrolled)'].sum())
#print('Taxa de aprovação 2° semestre:', 
#      data['Curricular units 2nd sem (approved)'].sum()/data['Curricular units 2nd sem (enrolled)'].sum())

data['Average grade per year'] = (data['Curricular units 1st sem (grade)'] + data['Curricular units 2nd sem (grade)']) / 2
data['Target'] = data['Target'].replace({'Enrolled': 0, 'Dropout': 1, 'Graduate': 2}) # doing ts for future heatmap

# Histogram mean grades 1st and 2nd semester

fig1, ax1_fig1 = plt.subplots(figsize=(16,9), num='Histograma da média das notas (1° e 2° sem)')
ax1_fig1.set_yticks(range(0, 4500, 100))
standard_ax_config(ax1_fig1)

ax1_fig1.hist(data['Curricular units 1st sem (grade)'], bins=range(0, 20), edgecolor='black', color='red', alpha=0.5, label='Média 1° sem')
ax1_fig1.hist(data['Curricular units 2nd sem (grade)'], bins=range(0, 20), edgecolor='black', color='skyblue', alpha=0.5, label='Média 2° sem')
ax1_fig1.legend()

# Do not need to worry about range excluding 20 because
# maximum value in dataset is < 19.

# Based on the grading system being 0-20, it is safe to assume
# the minimum grade required for passing is 10 (thanks google)

print(data['Curricular units 1st sem (grade)'].describe())
print(data['Curricular units 2nd sem (grade)'].describe())

# Since histogram has almost no values on 1 < x < 10, let's
# separate them in 3 groups: enrolled, dropout and success
# and then proceed our analysis

enrolled = data[data['Target'] == 0]
dropout  = data[data['Target'] == 1]
graduate  = data[data['Target'] == 2] 

fig2, (ax1_fig2) = plt.subplots(figsize=(12,6))

boxplot = ax1_fig2.boxplot((enrolled['Average grade per year'],
                  dropout['Average grade per year'],
                  graduate['Average grade per year']),
                  labels=['Enrolled', 'Dropout', 'Graduate'],
                  medianprops={'color': 'black', 'linewidth': '2'},
                  meanprops={'marker':'x'},
                  showmeans=True,
                  patch_artist=True)

for patch, color in zip(boxplot['boxes'], ['#FFD700', '#FF6347', '#87CEFA']):
    patch.set_facecolor(color)

plt.show()