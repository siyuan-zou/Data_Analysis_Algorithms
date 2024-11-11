import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.precision', 2)

data = pd.read_csv('csv/wines.csv', index_col=0)

print(data)

# stats per country
print('\nCountries:')
meanvar = pd.concat([data.mean(axis=1), data.var(axis=1, ddof=0), data.var(axis=1, ddof=1)], axis=1)
meanvar.columns = ['mean', 'variance', 'sample variance']
print(meanvar)

# correlation matrices
print('\nWines correlation matrix:')
print(data.corr())

print('\nCountries correlation matrix:')
print(data.transpose().corr())