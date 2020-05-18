
import os
from absl import app
from absl import logging
from absl import flags
from learner import *
from bandit import BernoulliBandit
from utils import draw, write_to_file

FLAGS = flags.FLAGS

# Flags for general settings
flags.DEFINE_string('out', 'data.out', 'file for generated data')
flags.DEFINE_string('fig', 'figure.png', 'file for generated figure')
flags.DEFINE_boolean('novar', True, 'do not show std in the output figure')
flags.DEFINE_boolean('rm', False, 'remove previously generated data')
flags.DEFINE_boolean('minimax', False, 'compute minimax regret based on given arms')
flags.DEFINE_float('prob', 0.2, 'the true probability of the first arm')
flags.DEFINE_integer('T', 1000, 'time horizon')
flags.DEFINE_integer('trials', 100, 'total number of trials')
flags.DEFINE_integer('freq', 50, 'frenquency to report the intermediate regrets')

# Flags for hyper-parameters
flags.DEFINE_float('eps', 1, 'parameter epsilon for epsilon greedy algorithm')
flags.DEFINE_float('C', 1, 'parameter C for explore-then-commit algorithm')
flags.DEFINE_float('alpha', 0.5, 'parameter alpha for UCB algorithm')


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

    # # If you want to generate reproducing result, uncomment the line below
    np.random.seed(200)

    # Extract all flags of parameters for later use
    trials = FLAGS.trials
    freq = FLAGS.freq
    T = FLAGS.T
    eps = FLAGS.eps
    C = FLAGS.C
    alpha = FLAGS.alpha
    prob = FLAGS.prob

    # Policies to be compared
    policies = [Greedy(), EpsGreedy(eps), ExploreThenCommit(C), BerUCB(alpha), TS()]
    # policies = [UCB(alpha)]

    # Get true distribution of Bernoulli arms
    mus = [(prob, 1 - prob)]
    if FLAGS.minimax:
        mus = [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.25, 0.75), (0.35, 0.65)]

    # The main loop
    for policy in policies:
        logging.info('run policy %s' % policy.name)
        for trial in range(trials):
            if trial % 50 == 0:
                logging.info('trial: %d' % trial)

            minimax_regret = dict()

            for (mu_1, mu_2) in mus:
                bernoulli_bandit = BernoulliBandit(mu_1, mu_2)
                agg_regret = dict()

                # The reason to exclude ExploreThenCommit is that it's not progressive.
                # Progressive means process of small T is nested in the process of large T
                if policy.name == 'ExploreThenCommit':
                    for comp_t in range(0, T + 1):
                        if comp_t % freq == 0:
                            if comp_t == 0:
                                agg_regret[comp_t] = 0
                            else:
                                # initialization
                                bernoulli_bandit.init()
                                policy.init(comp_t)
                                rewards = 0
                                for t in range(1, comp_t + 1):
                                    action = policy.choice(t)
                                    reward = bernoulli_bandit.pull_arm(action)
                                    policy.update(reward, action)
                                    rewards += reward
                                agg_regret[comp_t] = bernoulli_bandit.regret(rewards)
                else:
                    # initialization
                    bernoulli_bandit.init()
                    policy.init()
                    rewards = 0
                    for t in range(0, T + 1):
                        if t > 0:
                            action = policy.choice(t)
                            reward = bernoulli_bandit.pull_arm(action)
                            policy.update(reward, action)
                            rewards += reward
                        if t % freq == 0:
                            agg_regret[t] = bernoulli_bandit.regret(rewards)

                for t in agg_regret:
                    minimax_regret[t] = max(minimax_regret.get(t, 0), agg_regret[t])

            # output one trial result of into the output file
            write_to_file(dict({policy.name: minimax_regret}))

    # Generate the final figure
    draw()


# The main entry
if __name__ == '__main__':
    app.run(main)
