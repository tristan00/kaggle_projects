import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


path = r'C:\Users\trist\Documents\champs-scalar-coupling'

df_magnetic_shielding_tensors = pd.read_csv('{0}/{1}'.format(path, 'magnetic_shielding_tensors.csv'))
df_mulliken_charges = pd.read_csv('{0}/{1}'.format(path, 'mulliken_charges.csv'))
df_potential_energy = pd.read_csv('{0}/{1}'.format(path, 'potential_energy.csv'))
df_scalar_coupling_contributions = pd.read_csv('{0}/{1}'.format(path, 'scalar_coupling_contributions.csv'))
df_structures = pd.read_csv('{0}/{1}'.format(path, 'structures.csv'))
df_train = pd.read_csv('{0}/{1}'.format(path, 'train.csv'))
df_test = pd.read_csv('{0}/{1}'.format(path, 'test.csv'))

df_train = df_train[['atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant']]

train, val = train_test_split(df_train, test_size=.1, random_state=1)
ohe = OneHotEncoder()
ohe.fit(train['type'])

train_x = pd.get_dummies(train['atom_index_0', 'atom_index_1', 'type'])
train_y = train['scalar_coupling_constant']
val_x = pd.get_dummies(train['atom_index_0', 'atom_index_1', 'type'])
model = RandomForestRegressor()
model.fit(train_x, train_y)
print(model.predict())




