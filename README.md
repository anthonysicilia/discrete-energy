# discrete-energy
This is the package for computing the discrete energy distance as seen in the AACL2022 paper [LEATHER: A Framework for Learning to Generate Human-like Text in Dialogue](https://arxiv.org/pdf/2210.07777v1.pdf). Code is adapted from the original [code repo](https://github.com/anthonysicilia/LEATHER-AACL2022) for this paper. If you use this package, please consider citing the paper.

# Installation
To install this package, run the following command: 
```pip install git+https://github.com/anthonysicilia/classifier-divergence```

# Dependencies
This package was built and tested on Python 3.7.6. The requirements file has newer package numbers, but the code was also tested with the following versions:
 - torch==1.10.2 (build py3.7_cuda10.2_cudnn7.6.5_0)
 - scikit_learn==0.21.2

# Example
This shows a few ways to compute discrete energy on example data with the default or a custom coarsening function.

```
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
```
The output should look like this:
```
Energy (Option 1): 0.0158
Energy (Option 2): 0.0158
sqrt(energy): 0.1258 (better for comparison to test-divergence)
```
# More Papers
This package is part of series of works from our lab using learning theory to study understanding and generation in NLP. Check out some of our other papers/packages here:
- [The Change that Matters in Discourse Parsing: Estimating the Impact of Domain Shift on Parser Error](https://arxiv.org/abs/2203.11317)
- [PAC-Bayesian Domain Adaptation Bounds for Multiclass Learners](https://openreview.net/forum?id=S0lx6I8j9xq)
- [Modeling Non-Cooperative Dialogue: Theoretical and Empirical Insights](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00507/113020/Modeling-Non-Cooperative-Dialogue-Theoretical-and)
- [LEATHER: A Framework for Learning to Generate Human-like Text in Dialogue](https://arxiv.org/abs/2210.07777)
- [classifer-divergence](https://github.com/anthonysicilia/classifier-divergence)