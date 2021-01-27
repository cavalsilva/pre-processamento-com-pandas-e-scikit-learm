import pandas as pd

# Comando do pandas para exibir todas as colunas
pd.set_option('display.max_columns', None)

base = pd.read_csv('credit_data.csv')
base.describe()

# Lista os clientes com age negativo
base.loc[base['age'] < 0]

# Apagar os registros com problemas