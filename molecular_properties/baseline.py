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


def naive_use_other_data(df_main, df_other, base_name):
    df_other_1 = df_other.copy()
    df_other_2 = df_other.copy()
    df_other_1['atom_index_0'] = df_other_1['atom_index']
    df_other_2['atom_index_1'] = df_other_2['atom_index']
    df_other_1.columns = [
        i if i in ['atom_index_0', 'molecule_name'] else '{0}_1_'.format(base_name) + str(i) for i in
        df_other_1.columns]
    df_other_2.columns = [
        i if i in ['atom_index_1', 'molecule_name'] else '{0}_2_'.format(base_name) + str(i) for i in
        df_other_2.columns]
    df_main = df_main.merge(df_other_1)
    df_main = df_main.merge(df_other_2)
    return df_main


def use_potential_energy(df, df_2):
    df = df.merge(df_2)
    return df

print(df_train.shape)
df_train = naive_use_other_data(df_train, df_magnetic_shielding_tensors, 'magnetic_shielding_tensors')
print(df_train.shape)

df_train = naive_use_other_data(df_train, df_mulliken_charges, 'mulliken_charges')
print(df_train.shape)

df_train = naive_use_other_data(df_train, df_structures, 'structures')
print(df_train.shape)

df_train = use_potential_energy(df_train, df_potential_energy)
print(df_train.shape)


# df_test = naive_use_other_data(df_test, df_magnetic_shielding_tensors, 'magnetic_shielding_tensors')
# df_test = naive_use_other_data(df_test, df_mulliken_charges, 'mulliken_charges')
# df_test = naive_use_other_data(df_test, df_structures, 'structures')
# df_test = use_potential_energy(df_test, df_potential_energy)

df_train = df_train.dropna(subset = ['scalar_coupling_constant'])
df_train = df_train.drop(['id', 'molecule_name'], axis = 1)

train, val = train_test_split(df_train, test_size=.1, random_state=1)
train = train.reset_index()
val = val.reset_index()
null_columns1=train.columns[train.isnull().any()]

cols_to_drop = []
train_dfs_to_combine = []
val_dfs_to_combine = []
for i in train.columns:
    print(i, train[i].dtype)

    if train[i].dtype == object:
        cols_to_drop.append(i)
        ohe = OneHotEncoder()
        ohe.fit(train[i].values.reshape(-1, 1))
        transformed_train_df = pd.DataFrame(data = ohe.transform(train[i].values.reshape(-1, 1)).toarray(),
                                            columns = [str(i) + '_' + str(j) for j in ohe.categories_[0]])
        transformed_val_df = pd.DataFrame(data=ohe.transform(val[i].values.reshape(-1, 1)).toarray(),
                                            columns=[str(i) + '_' + str(j) for j in ohe.categories_[0]])
        train_dfs_to_combine.append(transformed_train_df)
        val_dfs_to_combine.append(transformed_val_df)

null_columns2=train.columns[train.isnull().any()]

train = train.drop(cols_to_drop, axis = 1)
val = val.drop(cols_to_drop, axis = 1)

train_dfs_to_combine.append(train)
val_dfs_to_combine.append(val)

train = pd.concat(train_dfs_to_combine, axis = 1)
val = pd.concat(val_dfs_to_combine, axis = 1)

null_columns3=train.columns[train.isnull().any()]


train_x = train.drop('scalar_coupling_constant', axis = 1)
val_x = val.drop('scalar_coupling_constant', axis = 1)
train_x = train_x.fillna(train_x.median())
val_x = val_x.fillna(train_x.median())

train_y = train['scalar_coupling_constant']
val_y = val['scalar_coupling_constant']

null_columns4=train_x.columns[train_x.isnull().any()]

rf = RandomForestRegressor()
rf.fit(train_x, train_y)
print(mean_squared_error(rf.predict(val_x), val_y))
