import json
from absl import logging
from absl import flags
import numpy as np
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

FLAGS = flags.FLAGS


class EmArm:
    """
    The class for storing empirical information of an arm
    """

    def __init__(self):
        self.reset()

    @property
    def rewards(self):
        return self.__rewards

    @property
    def pulls(self):
        return self.__pulls

    @property
    def good_pulls(self):
        return self.__good_pulls

    @property
    def em_mean(self):
        """get empirical mean"""
        if self.__pulls == 0:
            raise Exception('No empirical mean yet!')
        return self.__rewards / self.__pulls

    def reset(self):
        """clear historical records"""
        self.__pulls = 0
        self.__rewards = 0
        self.__good_pulls = 0

    def update(self, reward, good_pulls):
        self.__pulls += 1
        self.__rewards += reward
        if good_pulls:
            self.__good_pulls += 1


def draw():
    # read results of trials from file FLAGS.out, calculate average empirical
    # regret for each policy and draw the final figure
    col_learners = []
    col_horizons = []
    col_regrets = []
    with open(FLAGS.out, 'r') as f:
        for line in f:
            one_trial = json.loads(line)
            (learner, regrets) = list(one_trial.items())[0]
            for horizon in regrets:
                col_learners.append(learner)
                col_horizons.append(int(horizon))
                col_regrets.append(regrets[horizon])

    df = pd.DataFrame({'learner': col_learners, 'horizon': col_horizons,
                       'regret': col_regrets})

    if FLAGS.novar:
        ci_val = None
    else:
        ci_val = 'sd'

    ax = sns.lineplot(
        x='horizon', y='regret', hue='learner', data=df, ci=ci_val)

    # hide edges of filled area
    for child in ax.findobj(PolyCollection):
        child.set_linewidth(0.0)

    plt.ylabel('regret', fontweight='bold', fontsize=15)
    plt.xlabel('horizon', fontweight='bold', fontsize=15)
    logging.info('output figure to %s' % FLAGS.fig)
    plt.savefig(FLAGS.fig, format='png')
    plt.close()


def write_to_file(data):
    # write one trial result to file FLAGS.out
    with open(FLAGS.out, 'a') as f:
        json.dump(data, f)
        f.write('\n')
        f.flush()


def sphere_sampling(dim, samples):
    v = np.random.normal(size=(samples, dim))
    return v / np.sqrt(np.sum(v ** 2, 0))
