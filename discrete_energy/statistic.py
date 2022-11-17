import torch
from sklearn.cluster import KMeans

class DiscreteEnergyStatistic:

    def __init__(self, n_1, n_2, learn_clusters=True, nclusters=100, 
        seed=0, train_sample_1=None, train_sample_2=None):
        """
        Computes discrete energy between two samples.
        Class design based on the following repo:
        https://github.com/josipd/torch-two-sample

        Parameters
        ----------
        n_1: Sample size of first sample (test sample, if learn_clusters=True)
        n_2: Sample size of second sample (test sample, if learn_clusters=True)
        learn_clusters: whether to learn clusters (kmeans) or assume given
        seed: the seed to use for random processes when (if learn_clusters=True)
        train_sample_1: train set for first sample to learn clusters (if learn_clusters=True)
        train_sample_2: train set for second sample to learn clusters (if learn_clusters=True)

        Returns
        -------
        Initialized class
        """

        self.n_1 = n_1
        self.n_2 = n_2

        self.a00 = -1. / (n_1 * n_1)
        self.a11 = -1. / (n_2 * n_2)
        self.a01 = 1. / (n_1 * n_2)

        self.learn_clusters = learn_clusters
        if self.learn_clusters:
            if train_sample_1 is None or train_sample_2 is None:
                raise ValueError('When learn_clusters is True'
                    ' train_sample_1 and *_2 must be provided.')
            train_sample_12 = torch.cat((train_sample_1, train_sample_2), 0)
            coarsening_fn = KMeans(n_clusters=nclusters,
                random_state=seed).fit(train_sample_12)
            self.coarsening_fn = coarsening_fn.predict

    def __call__(self, sample_1, sample_2):
        
        

        sample_12 = torch.cat((sample_1, sample_2), 0)
        if self.learn_clusters:
            cats = torch.tensor(self.coarsening_fn(sample_12))
        else:
            cats = sample_12
        cats = cats.expand(cats.size(0), cats.size(0))
        cdist = (cats != cats.transpose(0, 1)).float()
        for i in range(cdist.size(0)): cdist[i, i] = 0
        d_1 = cdist[:self.n_1, :self.n_1].sum()
        d_2 = cdist[-self.n_2:, -self.n_2:].sum()
        d_12 = cdist[:self.n_1, -self.n_2:].sum()

        loss = 2 * self.a01 * d_12 + self.a00 * d_1 + self.a11 * d_2

        return loss