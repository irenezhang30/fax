from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
#
import hypothesis.extra.numpy

import jax.test_util
import jax.numpy as np
from jax import random
from jax import tree_util
from jax.experimental import optimizers
from jax.scipy.special import logsumexp
from jax.experimental.stax import softmax
from jax.config import config

from fax import converge
from fax import test_util
from fax.constrained import make_lagrangian
from fax.constrained import cga_lagrange_min
from fax.constrained import cga_ecp
from fax.constrained import slsqp_ecp
from fax.constrained import implicit_ecp

from exact_pg import policy_evaluation

config.update("jax_enable_x64", True)
config.update('jax_disable_jit', True)

# finding reward function
class CGATest(jax.test_util.JaxTestCase):

    # @parameterized.parameters(
    #     {'method': implicit_ecp,
    #      'kwargs': {'max_iter': 1000, 'lr_func': 0.01, 'optimizer': optimizers.adam}},
    #     {'method': cga_ecp, 'kwargs': {'max_iter': 1000, 'lr_func': 0.15, 'lr_multipliers': 0.925}},
    #     {'method': slsqp_ecp, 'kwargs': {'max_iter': 1000}},
    # )


    def test_omd(self):
    # def test_omd(self, method, kwargs):
        true_transition = np.array([[[0.7, 0.3], [0.2, 0.8]],
                                    [[0.99, 0.01], [0.99, 0.01]]])
        true_reward = np.array(([[-0.45, -0.1],
                                 [0.5,  0.5]]))
        temperature = 1e-2
        true_discount = 0.9
        initial_distribution = np.ones(2)/2

        policy_expert = np.array(([[0.4, 0.6],
                                   [0.4,  0.6]]))

        def smooth_bellman_optimality_operator(x, params):
            transition, reward, discount, temperature = params
            return reward + discount * np.einsum('ast,t->sa', transition, temperature *
                                                 logsumexp((1. / temperature) * x, axis=1))

        # @jax.jit
        def objective(x, params):
            del params
            policy = softmax((1. / temperature) * x) # [2, 2]
            cumulent = np.log(np.einsum('sa,ast->sat', policy, true_transition))
            cumulent = np.einsum('sat,ast->sa', cumulent, true_transition)
            likelihood = policy_evaluation(true_transition, cumulent, true_discount, policy_expert)
            print("policy", policy)
            return initial_distribution @ likelihood


        # @jax.jit
        def equality_constraints(x, params):
            #reward
            reward_logits = params
            reward_hat = softmax((1./temperature)*reward_logits)
            params = (true_transition, reward_hat, true_discount, temperature)
            return smooth_bellman_optimality_operator(x, params) - x

        initial_values = (
            np.zeros_like(true_reward),
            (np.zeros_like(true_reward))
        )

        args = {'max_iter': 1000}
        solution = slsqp_ecp(objective, equality_constraints, initial_values, **args)
        print ("solution", solution)


if __name__ == "__main__":
    absltest.main()

