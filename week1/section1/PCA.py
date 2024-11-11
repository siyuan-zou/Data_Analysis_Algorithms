#!/usr/bin/env python3

import sys
import pandas as pd

# matplotlib should be installed with
# pip3 install --user --only-binary=:all: matplotlib
# pip3 install --user --only-binary=:all: scikit-learn
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
pd.set_option('display.precision', 2)

def main():
    # usage message
    if len(sys.argv) != 2:  # we require exactly 1 argument
        print(f'Usage: {sys.argv[0]} csv_file')
        sys.exit(1)

    # read in data
    data = pd.read_csv(sys.argv[1], index_col=0)
    print(f'Read in the ({data.shape[0]}, {data.shape[1]}) matrix:')
    print(data)

    # stats per country
    print('\nCountries:')
    country_stats = pd.concat([data.mean(axis=1), data.var(axis=1, ddof=0), data.var(axis=1, ddof=1)], axis=1)
    country_stats.columns = ['mean', 'variance', 'sample variance']
    print(country_stats)

    # stats per wine
    print('\nWines:')
    wine_stats = pd.concat([data.mean(axis=0), data.var(axis=0, ddof=0), data.var(axis=0, ddof=1)], axis=1)
    wine_stats.columns = ['mean', 'variance', 'sample variance']
    print(wine_stats)

    # correlation matrices
    print('\nWines correlation matrix:')
    print(data.corr())

    print('\nCountries correlation matrix:')
    print(data.transpose().corr())

    # Histogram
    print("\n Plotting histogram; you have to close the plot to continue...", end='')
    fig, ax = plt.subplots()
    prev = [0] * data.shape[1]
    for index, country in data.iterrows():
        ax.bar(x=range(data.shape[1]),
               tick_label=data.columns.to_list(),
               height=country.values,
               label=index,
               bottom=prev)
        prev += country.values
    plt.xticks(rotation = 45)
    ax.set_ylabel('Consumption')
    ax.legend()
    plt.savefig("histogram.png")  # use "display histogram.png to open the plot from a Terminal
    print(" done")

    # normalize data using Sklearn
    print("\n Normalizing variances of individual variables...", end='')
    scaler = StandardScaler()
    ndata = scaler.fit_transform(data)
    print(" done")

    # PCA plot
    print("\n Plotting PCA; you have to close the plot to continue...", end='')
    pca = decomposition.PCA()
    pca.fit(ndata)
    X = pca.transform(ndata)
    plt.figure()
    plt.title("PCA of individuals")
    plt.grid(zorder=0.5, color='lightgrey')
    plt.scatter(X[:, 0], X[:, 1],zorder=2.5)
    for i, label in enumerate(data.transpose().columns):
        if i==0 or i==3 or i==4 or i==5: eps = 0.12
        else: eps = -0.35
        plt.annotate(label, (X[i,0], X[i,1]), xytext=(X[i,0], X[i,1]+eps))

    plt.savefig("PCA.png")  # use "display PCA.png" to open the plot in a Terminal
    print(" done")


if __name__ == '__main__':  # this is executed when the script is called from the command line!
    main()