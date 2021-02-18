import pandas as pd

base = pd.read_csv('census.csv')

# Transformação de variáveis categóricas
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()

# LabelEncoder: transforma base categórica em número 
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1]) # label da coluna workclass
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3]) # label da coluna education
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5]) # label da coluna marital-status
previsores[:,6] = labelencoder_previsores.fit_transform(previsores[:,6]) # label da coluna occupation
previsores[:,7] = labelencoder_previsores.fit_transform(previsores[:,7]) # label da coluna ralationship
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8]) # label da coluna race
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9]) # label da coluna sex
previsores[:,13] = labelencoder_previsores.fit_transform(previsores[:,13]) # label da coluna native_country

# Usar técnica de variável dummy, pois a utilização acima não é correta pois as
# variáveis acima não possui uma ordenação (não são mensuráveis), usando OneHotEncoder
onehotenconder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotenconder.fit_transform(previsores).toarray()

# Fazer o encoder da variável classe
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

# Escalonamento de valores/atributos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)