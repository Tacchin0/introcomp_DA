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

# grades and scholarship holder/tuites fees up do date
# seem to be correlated. change gdp, inflation and unemployement
# rates with variables mentioned

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

data['Average grade per year'] = (
    data['Curricular units 1st sem (grade)'] +
    data['Curricular units 2nd sem (grade)']
) / 2

data['Target'] = data['Target'].replace({
    'Enrolled': 0,
    'Dropout': 1,
    'Graduate': 2
})


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


# -------------------- GENERAL ANALYSIS (Vinicius) --------------------
#Grafico de pizza
fig_vinicius1, ax_vinicius1 = plt.subplots(figsize=(8, 8), num="Distribuição do Target")

target_counts = data['Target'].value_counts().sort_index()
labels = ['Matriculado', 'Evadido', 'Formado']
colors = ['#FFD700', '#FF6347', '#87CEFA']

ax_vinicius1.pie(
    target_counts, 
    labels=labels, 
    autopct='%1.1f%%', 
    colors=colors, 
    startangle=90, 
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.2}
)
ax_vinicius1.set_title("Distribuição do Status dos Alunos", fontsize=14, pad=20, fontweight='bold')

#Filtro E.M
dados_ensino_medio = data[data['Age at enrollment'] <= 19]

fig_vinicius2, ax_vinicius2 = plt.subplots(figsize=(10, 6), num="Idade vs Status (Ensino Médio)")

# Boxplot
sns.boxplot(
    data=dados_ensino_medio, 
    x='Target', 
    y='Age at enrollment', 
    palette=colors, 
    width=0.4,
    linewidth=1.5,
    ax=ax_vinicius2
)

# Adicionando pontos de dispersao
sns.stripplot(
    data=dados_ensino_medio, 
    x='Target', 
    y='Age at enrollment', 
    color='black', 
    alpha=0.35, 
    jitter=0.18, 
    size=5,
    ax=ax_vinicius2
)

#Ajustes e deixando mais bontio
ax_vinicius2.spines["top"].set_visible(False)
ax_vinicius2.spines["right"].set_visible(False)
ax_vinicius2.set_yticks(range(16, 21))
ax_vinicius2.set_ylim(15.5, 19.5)
ax_vinicius2.grid(axis='y', linestyle='--', alpha=0.5)
ax_vinicius2.set_axisbelow(True)
ax_vinicius2.set_xticklabels(labels, fontsize=11)
ax_vinicius2.set_title("Distribuição de Idade por Status Acadêmico (Até 19 anos)", fontsize=14, pad=15, fontweight='bold')
ax_vinicius2.set_xlabel("Status Acadêmico", fontsize=12, labelpad=10)
ax_vinicius2.set_ylabel("Idade na Matrícula (Anos)", fontsize=12, labelpad=10)
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

# Possivel grafico de barras -> média das notas por pagamento de mensalidade

tuiton_yes = data[data['Tuition fees up to date'] == 1]
tuiton_no = data[data['Tuition fees up to date'] == 0]

figAux, ax1_figAux = plt.subplots(figsize=(10, 6), num="Média das Notas por Mensalidade em Dia")

boxplotfigAux = ax1_figAux.boxplot((tuiton_yes["Average grade per year"],
                                    tuiton_no["Average grade per year"]),
                  tick_labels=['Sim', 'Não'],
                  medianprops={'color': 'black', 'linewidth': '2'},
                  meanprops={'marker':'x'},
                  showmeans=True,
                  patch_artist=True)

for patch, color in zip(boxplotfigAux['boxes'], ["#54C51C", "#EA0F0F"]):
    patch.set_facecolor(color)

ax1_figAux.set_title("Média das Notas por Mensalidade em Dia")
ax1_figAux.set_xlabel("Mensalidade em Dia")
ax1_figAux.set_ylabel("Média das Notas")

# Grafico 5 (de dispersão) -> como a bolsa de estudos afeta o desempenho e retenção dos alunos

fig5, ax1_fig5 = plt.subplots(figsize=(10, 6), num="Desempenho no 1° Semestre por Bolsa de Estudos")

data['Curricular units 1st sem (grade)'] = pd.to_numeric(
    data['Curricular units 1st sem (grade)'], errors='coerce'
)
bolsa = data['Scholarship holder'].map({0: 'Sem Bolsa', 1: 'Com Bolsa'})
target_hue = data['Target'].map({0: 'Matriculado', 1: 'Evadido', 2: 'Formado'})

sns.stripplot( data=data, x=bolsa, y='Curricular units 1st sem (grade)', hue=target_hue, jitter=True, alpha=0.6,
    palette={'Evadido': '#FF6347', 'Formado': '#87CEFA', 'Matriculado': '#FFD700'}, ax=ax1_fig5)

ax1_fig5.spines["top"].set_visible(False)
ax1_fig5.spines["right"].set_visible(False)
ax1_fig5.legend(title="Situação")

ax1_fig5.set_title("Desempenho dos Alunos no 1° Semestre por Bolsa de Estudos e Status Acadêmico")
ax1_fig5.set_xlabel("Bolsa de Estudos")
ax1_fig5.set_ylabel("Nota 1° Semestre")
# Transition matrix

fig6, (ax1_fig6) = plt.subplots(figsize=(4,3), num='Matriz de transição de desempenho')

ordem = ['Baixo', 'Médio', 'Alto']
categorizar = lambda x: ordem[0] if x < 10 else (ordem[1] if x < 15 else ordem[2])
data['Cat_1sem'] = data['Curricular units 1st sem (grade)'].apply(categorizar)
data['Cat_2sem'] = data['Curricular units 2nd sem (grade)'].apply(categorizar)

transicao = pd.crosstab(data['Cat_1sem'], data['Cat_2sem'])

ax1_fig6.set_xticks(range(len(ordem)))
ax1_fig6.set_xticklabels(ordem)
ax1_fig6.set_yticks(range(len(ordem)))
ax1_fig6.set_yticklabels(ordem)
ax1_fig6.set_xlabel('2º Semestre')
ax1_fig6.set_ylabel('1º Semestre')

for i in range(len(ordem)):
    for j in range(len(ordem)):
        ax1_fig6.text(j, i, str(transicao.iloc[i, j]), va='center', ha='center')

ax1_fig6.matshow(transicao, cmap='Blues')

# Correlation matrix

fig7, (ax1_fig7) = plt.subplots(figsize=(16,9), num='Heatmap de correlação')
fig7.subplots_adjust(bottom=0.3, left=0.20)
ax1_fig7.set_xticklabels(ax1_fig7.get_xticklabels(), fontsize=8)
ax1_fig7.set_yticklabels(ax1_fig7.get_yticklabels(), fontsize=8)
sns.heatmap(data.drop(columns=['Cat_1sem', 'Cat_2sem']).corr(), 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1, 
            ax=ax1_fig7)
plt.show()