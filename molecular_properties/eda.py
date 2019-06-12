import pandas as pd


path = r'C:\Users\trist\Documents\champs-scalar-coupling'


df_dipole_moments = pd.read_csv('{0}/{1}'.format(path, 'dipole_moments.csv'))
#define dipole movement
#are these 3 dim positions, if so investigate angles as features


df_magnetic_shielding_tensors = pd.read_csv('{0}/{1}'.format(path, 'magnetic_shielding_tensors.csv'))
#what are magnetic shielding tensors
#why are the repeated dims much larger than the others
#what is the index? atom num?

df_mulliken_charges = pd.read_csv('{0}/{1}'.format(path, 'mulliken_charges.csv'))
# define mulliken charges

df_potential_energy = pd.read_csv('{0}/{1}'.format(path, 'potential_energy.csv'))
#potential energy of what? energy released on molecule breakup?

df_scalar_coupling_contributions = pd.read_csv('{0}/{1}'.format(path, 'scalar_coupling_contributions.csv'))
#type, is this types of bonds?
# define fc, sd, pso, dso. Kaggle says they add up to the target.

df_structures = pd.read_csv('{0}/{1}'.format(path, 'structures.csv'))
# arn't positions of the locations arbitrary, may not be useful.

df_train = pd.read_csv('{0}/{1}'.format(path, 'train.csv'))


