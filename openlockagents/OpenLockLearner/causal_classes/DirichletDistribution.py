import numpy as np
import scipy


# todo: rethink how to do local/global attribute sampling.
class DirichletDistribution:
    def __init__(self, dimensionality, prior=None, initial_alpha=1):
        if prior is None:
            self.frequency_count = np.full(dimensionality, initial_alpha)
        else:
            # todo: should we initialize frequency count to prior - must be a less hacky way to do this?
            self.frequency_count = np.array(prior)
        self.prior = prior
        self.sampled_multinomial = None
        self.frequency_distribution = None
        self.renormalize()

    def __str__(self):
        return str(self.sampled_multinomial)

    def __repr__(self):
        return str(self)

    def update_alpha(self, index, alpha_increase=1):
        self.frequency_count[index] += alpha_increase
        self.renormalize()

    def renormalize(self):
        self.frequency_distribution = self.frequency_count / sum(self.frequency_count)
        self.sampled_multinomial = self.sample_multinomial()

    def sample_multinomial(self):
        return np.random.dirichlet(self.frequency_count)

    def sample_category(self):
        r = np.random.uniform(0, 1)
        sum = 0
        index = 0
        while sum < r:
            sum += self.frequency_distribution[index]
            index += 1
        return index

    def compute_confidence(self):
        """
        C[X] = 1 - (H[X] / H_{max}[X])
        :return:
        """
        max_entropy = scipy.stats.entropy(
            np.full(
                self.frequency_distribution.shape,
                1 / self.frequency_distribution.shape[0],
            )
        )
        # todo: compute information gain in a similar fashion
        entropy = scipy.stats.entropy(self.frequency_distribution)
        return 1 - (entropy / max_entropy)

    def convert_to_list(self):
        if isinstance(self.frequency_distribution, np.ndarray):
            self.frequency_distribution = self.frequency_distribution.tolist()
        if isinstance(self.frequency_count, np.ndarray):
            self.frequency_count = self.frequency_count.tolist()
        if self.prior is not None and isinstance(self.prior, np.ndarray):
            self.prior = self.prior.tolist()

    def pretty_distribution(self):
        return [float("{0:0.3f}".format(i)) for i in self.sampled_multinomial]
