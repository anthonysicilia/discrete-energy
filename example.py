import torch
from sklearn.cluster import KMeans
from math import sqrt
from discrete_energy.statistic import DiscreteEnergyStatistic

if __name__ == '__main__':

    # be careful not to make N too large,
    # the space/time complexity is O(N^2)
    N = 5_000
    # load example data; e.g., neural feature representations of text
    s1 = torch.load(open('example_data/sample1.pt', 'rb'))
    s2 = torch.load(open('example_data/sample2.pt', 'rb'))
    # split into train and test
    # the coarseninf function should be learned indep.
    # from the test set to avoid overfitting
    test_s1 = s1[:N]
    test_s2 = s2[:N]
    train_s1 = s1[N:2*N]
    train_s2 = s2[N:2*N]

    # Option 1: Learn default coarsening function (k-means)
    stat = DiscreteEnergyStatistic(N, N, learn_clusters=True,
        nclusters=100, seed=1, 
        train_sample_1=train_s1,
        train_sample_2=train_s2)
    print('Energy (Option 1):', f'{stat(test_s1, test_s2):.4f}')

    # Option 2: Learn your own coarsening function
    train_s12 = torch.cat((train_s1, train_s2), 0)
    coarsening_fn = KMeans(n_clusters=100, random_state=1).fit(train_s12)
    test_s1 = torch.tensor(coarsening_fn.predict(test_s1))
    test_s2 = torch.tensor(coarsening_fn.predict(test_s2))
    stat = DiscreteEnergyStatistic(N, N, learn_clusters=False)
    print('Energy (Option 2):', f'{stat(test_s1, test_s2):.4f}')

    # sqrt(energy) and test-divergence are linearly related in our empirical results
    print('sqrt(energy):', f'{sqrt(stat(test_s1, test_s2)):.4f}', '(better for comparison to test-divergence)')
