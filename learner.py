from abc import ABC, abstractmethod
from utils import EmArm
import numpy as np


class Learner(ABC):
    """
    Abstrct basic class (an interface) to be inherited
    """

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def choice(self, time):
        pass

    @abstractmethod
    def update(self, reward, action):
        pass


class Uniform_Sampling(Learner):
    """
    A test class to check everthing is ok by uniform sampling. Do use it in real
    learning process.
    """

    @property
    def name(self):
        return 'Uniform'

    def init(self, contexts):
        self.__contexts = contexts

    def choice(self, time):
        return time % len(self.__contexts)

    def update(self, reward, action):
        pass


class Greedy(Learner):
    """
    The class of naive greedy police (bernoulli arms only)
    """

    def __init__(self):
        self.__em_arm_1 = EmArm()
        self.__em_arm_2 = EmArm()

    @property
    def name(self):
        return 'Greedy'

    def init(self):
        self.__em_arm_1.reset()
        self.__em_arm_2.reset()

    def choice(self, time):
        if time == 1:
            return 1
        elif time == 2:
            return 2
        else:
            if self.__em_arm_1.em_mean >= self.__em_arm_2.em_mean:
                return 1
            else:
                return 2

    def update(self, reward, action):
        if action == 1:
            self.__em_arm_1.update(reward, reward == 1)
        else:
            self.__em_arm_2.update(reward, reward == 1)


class EpsGreedy(Learner):
    """
    The class of greedy policy with epsilon exception (bernoulli arms only)
    """

    def __init__(self, epsilon):
        self.__eps = epsilon
        self.__em_arm_1 = EmArm()
        self.__em_arm_2 = EmArm()

    @property
    def name(self):
        return 'Epsilon-Greedy'

    def init(self):
        self.__em_arm_1.reset()
        self.__em_arm_2.reset()

    def choice(self, time):
        if time == 1:
            return 1
        elif time == 2:
            return 2
        else:
            rnd_1 = np.random.random_sample(1)
            if rnd_1 <= (1 - self.__eps / time):
                if self.__em_arm_1.em_mean >= self.__em_arm_2.em_mean:
                    return 1
                else:
                    return 2
            else:
                rnd_2 = np.random.random_sample(1)
                if rnd_2 <= 0.5:
                    return 1
                else:
                    return 2

    def update(self, reward, action):
        if action == 1:
            self.__em_arm_1.update(reward, reward == 1)
        else:
            self.__em_arm_2.update(reward, reward == 1)


class ExploreThenCommit(Learner):
    """
    The class of explore-then-commit policy (bernoulli arms only)
    """

    def __init__(self, C):
        self.__C = C
        self.__em_arm_1 = EmArm()
        self.__em_arm_2 = EmArm()
        self.__trial_time = 0

    @property
    def name(self):
        return 'ExploreThenCommit'

    def init(self, total_time):
        self.__em_arm_1.reset()
        self.__em_arm_2.reset()
        self.__trial_time = self.__C * np.power(total_time, 2 / 3)

    def choice(self, time):
        if time <= np.ceil(self.__trial_time / 2):
            return 1
        elif (time > np.ceil(self.__trial_time / 2)) and (time <= np.ceil(self.__trial_time)):
            return 2

        else:
            if self.__em_arm_1.em_mean >= self.__em_arm_2.em_mean:
                return 1
            else:
                return 2

    def update(self, reward, action):
        if action == 1:
            self.__em_arm_1.update(reward, reward == 1)
        else:
            self.__em_arm_2.update(reward, reward == 1)


class BerUCB(Learner):
    """
    The class of Upper Confidence Bound (UCB) policy (bernoulli arms only)
    """

    def __init__(self, alpha):
        self.__alpha = alpha
        self.__em_arm_1 = EmArm()
        self.__em_arm_2 = EmArm()

    @property
    def name(self):
        return 'UCB'

    def init(self):
        self.__em_arm_1.reset()
        self.__em_arm_2.reset()

    def choice(self, time):
        if time == 1:
            return 1
        elif time == 2:
            return 2
        else:
            pulls_1 = self.__em_arm_1.pulls
            pulls_2 = self.__em_arm_2.pulls
            upper_1 = self.__em_arm_1.em_mean + self.__alpha * np.sqrt(2 * np.log(time - 1) / pulls_1)
            upper_2 = self.__em_arm_2.em_mean + self.__alpha * np.sqrt(2 * np.log(time - 1) / pulls_2)
            if upper_1 >= upper_2:
                return 1
            else:
                return 2

    def update(self, reward, action):
        if action == 1:
            self.__em_arm_1.update(reward, reward == 1)
        else:
            self.__em_arm_2.update(reward, reward == 1)


class TS(Learner):
    """
    The class of Thompson sampling policy (bernoulli arms only)
    """

    def __init__(self):
        self.__em_arm_1 = EmArm()
        self.__em_arm_2 = EmArm()

    @property
    def name(self):
        return 'TS'

    def init(self):
        self.__em_arm_1.reset()
        self.__em_arm_2.reset()

    def choice(self, time):
        good_pulls_1 = self.__em_arm_1.good_pulls
        good_pulls_2 = self.__em_arm_2.good_pulls
        bad_pulls_1 = self.__em_arm_1.pulls - self.__em_arm_1.good_pulls
        bad_pulls_2 = self.__em_arm_2.pulls - self.__em_arm_2.good_pulls

        est_1 = np.random.beta(1 + good_pulls_1, 1 + bad_pulls_1, 1)
        est_2 = np.random.beta(1 + good_pulls_2, 1 + bad_pulls_2, 1)

        if est_1 >= est_2:
            return 1
        else:
            return 2

    def update(self, reward, action):
        if action == 1:
            self.__em_arm_1.update(reward, reward == 1)
        else:
            self.__em_arm_2.update(reward, reward == 1)


class MultiUCB(Learner):
    """
    The class of Upper Confidence Bound (UCB) policy (multi-arms)
    """

    def __init__(self, alpha):
        self.__alpha = alpha

    @property
    def name(self):
        return 'UCB'

    def init(self, contexts):
        self.__dim = len(contexts)
        self.__em_arm = [EmArm() for i in range(self.__dim)]

    def choice(self, time):
        if time <= self.__dim:
            return time - 1

        else:
            max_recorder = [0, 0]
            for i in range(self.__dim):
                which_arm = self.__em_arm[i]
                upper = which_arm.em_mean + \
                        self.__alpha * np.sqrt(2 * np.log(time - 1) / which_arm.pulls)
                if upper >= max_recorder[1]:
                    max_recorder = [i, upper]
            return max_recorder[0]

    def update(self, reward, action):
        """No need to define good/bad pulls. Automatically good pulls entered"""
        self.__em_arm[action].update(reward, True)


class LinUCB(Learner):
    """
    This class is to learn linear contextual bandits
    """

    def __init__(self, alpha, T):
        self.__alpha = alpha
        self.__T = T

    @property
    def name(self):
        return 'LinUCB'

    def init(self, contexts):
        self.__contexts = contexts
        self.__dim = len(contexts)
        self.__A = np.eye(3)
        self.__b = np.zeros(3)

    def choice(self, time):
        A_inv = np.linalg.inv(self.__A)
        est_theta = A_inv @ self.__b

        max_recorder = [0, 0]
        for i in range(self.__dim):
            which_context = np.array(self.__contexts[i])
            upper = which_context @ est_theta + \
                    self.__alpha * np.sqrt(which_context @ A_inv @ which_context) * \
                    np.sqrt(np.log(self.__dim * self.__T ** 2))

            if upper >= max_recorder[1]:
                max_recorder = [i, upper]

        return max_recorder[0]

    def update(self, reward, action):
        which_context = np.array(self.__contexts[action])
        self.__A += np.outer(which_context, which_context)
        self.__b += reward * which_context
