import pandas as pd

# Comando do pandas para exibir todas as colunas
pd.set_option('display.max_columns', None)

base = pd.read_csv('credit_data.csv')
base.describe()

# Lista os clientes com age negativo
base.loc[base['age'] < 0]

'''
TRATAMENTO DE VALORES INCONSISTENTES
Preencher os valores de Age manualmente, será preenchido os valores com a média
das idades positivas
'''
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92

# Verificar se tem valor faltante em Age
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

'''
TRATAMENTO DE VALORES FALTANTES
Divivir a base de dados sendo Previsores e Classe:

Previsores: Às colunas Income, Age e Loan (a coluna clientId não é necessária,
pois não agrega para o modelo)

Classe: A última coluna que determinará se deve ou não realizar o empréstimo
'''
previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:,0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

'''
ESCALONAMENTO DE ATRIBUTOS
No caso abaixo o escalonamento deve ser feito para Income e Age, para que o
algoritmo trate os dois com a mesma importância
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Padronização por escala
previsores = scaler.fit_transform(previsores)