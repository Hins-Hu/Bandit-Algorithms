from absl import logging
import numpy as np


class BernoulliBandit():
    """
    The class of a bandit with Bernoulli arm indexed by 1 and 2
    """

    def __init__(self, mu_1, mu_2):
        self.__mu_1, self.__mu_2 = mu_1, mu_2

    def init(self):
        self.tot_samples = 0

    def pull_arm(self, index):
        if index not in [1, 2]:
            logging.fatal('Wrong Arm Index!')
        self.tot_samples += 1
        if index == 1:
            return np.random.binomial(1, self.__mu_1, 1)[0]
        return np.random.binomial(1, self.__mu_2, 1)[0]

    def regret(self, rewards):
        return self.tot_samples * max(self.__mu_1, self.__mu_2) - rewards


class GaussianArm():
    """
    gaussian arm
    """

    def __init__(self, mu, sigma=1):
        self.__mu = mu
        self.__sigma = sigma

    @property
    def mean(self):
        return self.__mu

    @property
    def std(self):
        return self.__sigma

    def pull(self):
        """return a numpy array of stochastic rewards"""
        return np.random.normal(self.__mu, self.__sigma, 1)[0]


class LinearBandit():
    """
    Linear Bandit Class
    Arms are numbered by 0 to len(contexts)-1 by default.
    """

    def __init__(self, contexts, theta):
        if len(contexts) < 2:
            logging.fatal('The number of arms should be at least two!')
        if not isinstance(contexts, list):
            logging.fatal('Features should be given in a list!')
        self.__contexts = [np.array(context) for context in contexts]
        self.__theta = np.array(theta)
        for _, context in enumerate(self.__contexts):
            if context.shape != self.__theta.shape:
                logging.fatal('The context and theta dimensions are unequal!')
        arms = [GaussianArm(np.dot(context, theta)) for context in self.__contexts]
        self.__arms = arms
        self.__arm_num = len(arms)
        self.__best_arm_ind = max([(tup[0], tup[1].mean) for tup in enumerate(self.__arms)], key=lambda x: x[1])[0]
        self.__best_arm = self.__arms[self.__best_arm_ind]

    def init(self):
        self.tot_samples = 0
        self.__max_rewards = 0

    def pull_arm(self, index):
        if index not in range(self.__arm_num):
            logging.fatal('Wrong arm index!')
        self.tot_samples += 1
        return self.__arms[index].pull()

    @property
    def all_arm(self):
        return self.__arms

    @property
    def contexts(self):
        return self.__contexts

    def regret(self, rewards):
        return self.tot_samples * self.__best_arm.mean - rewards
