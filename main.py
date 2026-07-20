import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
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
# bar graph for inflation/unemplyement rate and gradJe

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

#print('Taxa de aprovação 1° semestre:',
#      data['Curricular units 1st sem (approved)'].sum()/data['Curricular units 1st sem (enrolled)'].sum())
#print('Taxa de aprovação 2° semestre:',
#      data['Curricular units 2nd sem (approved)'].sum()/data['Curricular units 2nd sem (enrolled)'].sum())

data['Average grade per year'] = (
    data['Curricular units 1st sem (grade)'] +
    data['Curricular units 2nd sem (grade)']
) / 2

data['Target'] = data['Target'].replace({
    'Enrolled': 0,
    'Dropout': 1,
    'Graduate': 2
})

# -------------------- ECONOMIC ANALYSIS --------------------

# Média das notas por taxa de inflação
inflation_grade = (
    data.groupby("Inflation rate")["Average grade per year"]
    .mean()
    .sort_index()
)

fig3, ax1_fig3 = plt.subplots(
    figsize=(10, 6),
    num="Média das Notas por Taxa de Inflação"
)

# Não usamos standard_ax_config para gráficos de barras
ax1_fig3.spines["top"].set_visible(False)
ax1_fig3.spines["right"].set_visible(False)

ax1_fig3.bar(
    range(len(inflation_grade)),
    inflation_grade.values,
    color="steelblue",
    edgecolor="black",
)

ax1_fig3.set_xticks(range(len(inflation_grade)))
ax1_fig3.set_xticklabels(
    [f"{x:.1f}%" for x in inflation_grade.index]
)

ax1_fig3.set_ylim(9.5, 11.5)

ax1_fig3.set_title("Média das Notas por Taxa de Inflação")
ax1_fig3.set_xlabel("Taxa de Inflação (%)")
ax1_fig3.set_ylabel("Média das Notas")


# Média das notas por taxa de desemprego
unemployment_grade = (
    data.groupby("Unemployment rate")["Average grade per year"]
    .mean()
    .sort_index()
)

fig4, ax1_fig4 = plt.subplots(
    figsize=(10, 6),
    num="Média das Notas por Taxa de Desemprego"
)

ax1_fig4.spines["top"].set_visible(False)
ax1_fig4.spines["right"].set_visible(False)

ax1_fig4.bar(
    range(len(unemployment_grade)),
    unemployment_grade.values,
    color="tomato",
    edgecolor="black",
)

ax1_fig4.set_xticks(range(len(unemployment_grade)))
ax1_fig4.set_xticklabels(
    [f"{x:.1f}%" for x in unemployment_grade.index]
)

ax1_fig4.set_ylim(9.5, 11.5)

ax1_fig4.set_title("Média das Notas por Taxa de Desemprego")
ax1_fig4.set_xlabel("Taxa de Desemprego (%)")
ax1_fig4.set_ylabel("Média das Notas")

# Histogram mean grades 1st and 2nd semester

fig1, ax1_fig1 = plt.subplots(figsize=(16,9), num='Histograma da média das notas (1° e 2° sem)')
ax1_fig1.set_yticks(range(0, 4500, 100))
standard_ax_config(ax1_fig1)

ax1_fig1.hist(data['Curricular units 1st sem (grade)'], bins=range(0, 20), edgecolor='black', color='red', alpha=0.5, label='Média 1° sem')
ax1_fig1.hist(data['Curricular units 2nd sem (grade)'], bins=range(0, 20), edgecolor='black', color='skyblue', alpha=0.5, label='Média 2° sem')
ax1_fig1.legend()

print(data['Curricular units 1st sem (grade)'].describe())
print(data['Curricular units 2nd sem (grade)'].describe())

enrolled = data[data['Target'] == 0]
dropout  = data[data['Target'] == 1]
graduate = data[data['Target'] == 2]

fig2, ax1_fig2 = plt.subplots(figsize=(12,6))

boxplot = ax1_fig2.boxplot(
    (
        enrolled['Average grade per year'],
        dropout['Average grade per year'],
        graduate['Average grade per year']
    ),
    labels=['Enrolled', 'Dropout', 'Graduate'],
    medianprops={'color': 'black', 'linewidth': 2},
    meanprops={'marker': 'x'},
    showmeans=True,
    patch_artist=True
)

for patch, color in zip(boxplot['boxes'], ['#FFD700', '#FF6347', '#87CEFA']):
    patch.set_facecolor(color)

plt.show()
