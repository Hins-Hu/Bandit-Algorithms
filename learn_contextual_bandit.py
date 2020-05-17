
import os
from absl import app
from absl import logging
from absl import flags
from learner import Uniform_Sampling, MultiUCB, LinUCB
from bandit import LinearBandit
from utils import draw, write_to_file, sphere_sampling
import numpy as np

FLAGS = flags.FLAGS

# flag of general settings
flags.DEFINE_string('out', 'data.out', 'file for generated data')
flags.DEFINE_string('fig', 'figure.png', 'file for generated figure')
flags.DEFINE_boolean('novar', True, 'do not show std in the output figure')
flags.DEFINE_boolean('rm', False, 'remove previously generated data')
flags.DEFINE_integer('T', 1000, 'time horizon')
flags.DEFINE_integer('trials', 50, 'total number of trials')
flags.DEFINE_integer('freq', 50, 'frenquency to report the intermediate regrets')
flags.DEFINE_integer('armnum', 10, 'number of arms for for input')
flags.DEFINE_boolean('minimax', True, 'compute minimax regret')
flags.DEFINE_integer('inputnum', 10, 'number of inputs used in computing minimax regret')

# flag of hyperparameters
flags.DEFINE_float('alpha', 0.5, 'the hyper-parameter for UCB')
flags.DEFINE_float('alpha_LinUCB', 0.1, 'the hyper-parameter for LinUCB')


def main(argv):
    del argv

    if FLAGS.rm:
        os.remove(FLAGS.out)
    else:
        if FLAGS.out in os.listdir('./'):
            logging.fatal(('%s is not empty. Make sure you have'
                           ' archived previously generated data. '
                           'Try --rm flag which will automatically'
                           ' delete previous data.') % FLAGS.out)

    # for reproducing purpose
    # np.random.seed(100)

    trials = FLAGS.trials
    freq = FLAGS.freq
    T = FLAGS.T
    inputnum = FLAGS.inputnum if FLAGS.minimax else 1

    # policies to be compared
    # add your methods here
    policies = [MultiUCB(FLAGS.alpha), LinUCB(FLAGS.alpha_LinUCB, FLAGS.T)]

    for policy in policies:
        logging.info('run policy %s' % policy.name)
        for trial in range(trials):
            if trial % 50 == 0:
                logging.info('trial: %d' % trial)

            minimax_regret = dict()

            for _ in range(inputnum):
                contexts = list(sphere_sampling(3, FLAGS.armnum))
                theta = [1, 0, 0]
                bandit = LinearBandit(contexts, theta)
                agg_regret = dict()
                # initialization
                bandit.init()
                policy.init(contexts)
                rewards = 0
                for t in range(0, T + 1):
                    if t > 0:
                        action = policy.choice(t)
                        reward = bandit.pull_arm(action)
                        policy.update(reward, action)
                        rewards += reward
                    if t % freq == 0:
                        agg_regret[t] = bandit.regret(rewards)
                for t in agg_regret:
                    minimax_regret[t] = max(minimax_regret.get(t, 0), agg_regret[t])
            # output one trial result into the output file
            write_to_file(dict({policy.name: minimax_regret}))

    # generate the final figure
    draw()


if __name__ == '__main__':
    app.run(main)
