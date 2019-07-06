import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

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


def get_molecule_features():
    pass


def feature_generation(df_train, df_val, df_test,
                       df_magnetic_shielding_tensors,
                       df_mulliken_charges,
                       df_potential_energy,
                       df_scalar_coupling_contributions,
                       df_structures):
    df_train = naive_use_other_data(df_train, df_magnetic_shielding_tensors, 'magnetic_shielding_tensors')
    df_train = naive_use_other_data(df_train, df_mulliken_charges, 'mulliken_charges')
    df_train = naive_use_other_data(df_train, df_structures, 'structures')
    df_train = df_train.merge(df_potential_energy)
    df_train_alt = naive_use_other_data(df_train, df_scalar_coupling_contributions, 'scalar_coupling_contributions')

    df_val = naive_use_other_data(df_val, df_magnetic_shielding_tensors, 'magnetic_shielding_tensors')
    df_val = naive_use_other_data(df_val, df_mulliken_charges, 'mulliken_charges')
    df_val = naive_use_other_data(df_val, df_structures, 'structures')
    df_val = df_val.merge(df_potential_energy)
    df_val_alt = naive_use_other_data(df_val, df_scalar_coupling_contributions, 'scalar_coupling_contributions')

    cols_to_drop = []
    train_dfs_to_combine = []
    val_dfs_to_combine = []
    for i in df_train_alt.columns:
        print(i, df_train_alt[i].dtype)

        if df_train_alt[i].dtype == object:
            cols_to_drop.append(i)
            ohe = OneHotEncoder()
            ohe.fit(df_train_alt[i].values.reshape(-1, 1))
            transformed_train_df = pd.DataFrame(data=ohe.transform(df_train_alt[i].values.reshape(-1, 1)).toarray(),
                                                columns=[str(i) + '_' + str(j) for j in ohe.categories_[0]])
            transformed_val_df = pd.DataFrame(data=ohe.transform(df_val_alt[i].values.reshape(-1, 1)).toarray(),
                                              columns=[str(i) + '_' + str(j) for j in ohe.categories_[0]])
            train_dfs_to_combine.append(transformed_train_df)
            val_dfs_to_combine.append(transformed_val_df)

    train = df_train_alt.drop(cols_to_drop, axis=1)
    val = df_val_alt.drop(cols_to_drop, axis=1)

    train_dfs_to_combine.append(train)
    val_dfs_to_combine.append(val)

    train = pd.concat(train_dfs_to_combine, axis=1)
    val = pd.concat(val_dfs_to_combine, axis=1)


    valid_cols = None
    model_fc = get_model()



def get_model(input_shape):
    model = Sequential()
    model.add(Dense(512, input_dim=input_shape, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model


if __name__ == '__main__':
    path = r'C:\Users\trist\Documents\champs-scalar-coupling'

    df_magnetic_shielding_tensors = pd.read_csv('{0}/{1}'.format(path, 'magnetic_shielding_tensors.csv'))
    df_mulliken_charges = pd.read_csv('{0}/{1}'.format(path, 'mulliken_charges.csv'))
    df_potential_energy = pd.read_csv('{0}/{1}'.format(path, 'potential_energy.csv'))
    df_scalar_coupling_contributions = pd.read_csv('{0}/{1}'.format(path, 'scalar_coupling_contributions.csv'))
    df_structures = pd.read_csv('{0}/{1}'.format(path, 'structures.csv'))
    df_train = pd.read_csv('{0}/{1}'.format(path, 'train.csv'))
    df_test = pd.read_csv('{0}/{1}'.format(path, 'test.csv'))
    train, val = train_test_split(df_train, test_size=.1, random_state=1)
    train = train.reset_index()
    val = val.reset_index()
    feature_generation(train, val, df_test, df_magnetic_shielding_tensors,
                       df_mulliken_charges, df_potential_energy,
                       df_scalar_coupling_contributions, df_structures)


