import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join

# dataset link
# https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data
# check it out later for infos about categorical values
# https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

# TODO

# GENERAL (vinicius)
# add pizza graph for target variable
# maybe add scatter plot or boxplot the same for age and target
# improve graphs (exclusive for renato!!)

# ECONOMIC (pedro/luiz)
# bar graph for inflation/unemplyement rate and grade
# add scatter plot comparing some economic variable with mean grade/target

# MATRICES (renato)
# add correlation matrix
# check performance and retention rate using a transition matrix (search for crosstab)

# prazo: 22/07

def standard_ax_config(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

path = kagglehub.dataset_download("thedevastator/higher-education-predictors-of-student-retention")
data = pd.read_csv(join(path, 'dataset.csv'))

# No need for pre-processing since researchers stated
# it was already been done.

print(data)
data.info()

data['Average grade per year'] = (data['Curricular units 1st sem (grade)'] + data['Curricular units 2nd sem (grade)']) / 2
data['Target'] = data['Target'].replace({'Enrolled': 0, 'Dropout': 1, 'Graduate': 2}).infer_objects(copy=False)

# Histogram mean grades 1st and 2nd semester

fig1, (ax1_fig1) = plt.subplots(figsize=(12,6), num='Histograma da média das notas (1° e 2° sem)')
ax1_fig1.set_xlabel('Média das notas 1° e 2° sem')
ax1_fig1.set_ylabel('Número de alunos')
ax1_fig1.set_yticks(range(0, 4500, 100))
standard_ax_config(ax1_fig1)

ax1_fig1.hist(data['Curricular units 1st sem (grade)'],
              bins=range(0, 20),
              edgecolor='black',
              color='red',
              alpha=0.5,
              label='Média 1° sem')

ax1_fig1.hist(data['Curricular units 2nd sem (grade)'],
              bins=range(0, 20),
              edgecolor='black',
              color='skyblue',
              alpha=0.5,
              label='Média 2° sem')

ax1_fig1.legend()

# Do not need to worry about range excluding 20 because
# maximum value in dataset is < 19.

# Boxplot of these 3 status below

enrolled = data[data['Target'] == 0]
dropout  = data[data['Target'] == 1]
graduate = data[data['Target'] == 2]

fig2, (ax1_fig2) = plt.subplots(figsize=(12,6), num='Boxplot dos status dos estudantes')
ax1_fig2.set_ylabel("Média das notas 1° e 2° semestre")

boxplot = ax1_fig2.boxplot((enrolled['Average grade per year'],
                  dropout['Average grade per year'],
                  graduate['Average grade per year']),
                  tick_labels=['Enrolled', 'Dropout', 'Graduate'],
                  medianprops={'color': 'black', 'linewidth': '2'},
                  meanprops={'marker':'x'},
                  showmeans=True,
                  patch_artist=True)

for patch, color in zip(boxplot['boxes'], ['#FFD700', '#FF6347', '#87CEFA']):
    patch.set_facecolor(color)

# Transition matrix

fig3, (ax1_fig3) = plt.subplots(figsize=(4,3), num='Matriz de transição de desempenho')

ordem = ['Baixo', 'Médio', 'Alto']
categorizar = lambda x: ordem[0] if x < 10 else (ordem[1] if x < 15 else ordem[2])
data['Cat_1sem'] = data['Curricular units 1st sem (grade)'].apply(categorizar)
data['Cat_2sem'] = data['Curricular units 2nd sem (grade)'].apply(categorizar)

transicao = pd.crosstab(data['Cat_1sem'], data['Cat_2sem'])

ax1_fig3.set_xticks(range(len(ordem)))
ax1_fig3.set_xticklabels(ordem)
ax1_fig3.set_yticks(range(len(ordem)))
ax1_fig3.set_yticklabels(ordem)
ax1_fig3.set_xlabel('2º Semestre')
ax1_fig3.set_ylabel('1º Semestre')

for i in range(len(ordem)):
    for j in range(len(ordem)):
        ax1_fig3.text(j, i, str(transicao.iloc[i, j]), va='center', ha='center')

ax1_fig3.matshow(transicao, cmap='Blues')

# Correlation matrix

fig4, (ax1_fig4) = plt.subplots(figsize=(16,9), num='Heatmap de correlação')
fig4.subplots_adjust(bottom=0.3, left=0.20)
ax1_fig4.set_xticklabels(ax1_fig4.get_xticklabels(), fontsize=8)
ax1_fig4.set_yticklabels(ax1_fig4.get_yticklabels(), fontsize=8)
sns.heatmap(data.drop(columns=['Cat_1sem', 'Cat_2sem']).corr(), 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            ax=ax1_fig4)
plt.show()