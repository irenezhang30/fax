import jax
import jax.numpy as np
from jax import grad, jit
from jax.scipy.special import logsumexp


def dadashi_fig2d():
    """ Figure 2 d) of
    ''The Value Function Polytope in Reinforcement Learning''
    by Dadashi et al. (2019) https://arxiv.org/abs/1901.11524
    Returns:
        tuple (P, R, gamma) where the first element is a tensor of shape
        (A x S x S), the second element 'R' has shape (S x A) and the
        last element is the scalar (float) discount factor.
    """
    P = np.array([[[0.7, 0.3], [0.2, 0.8]],
                  [[0.99, 0.01], [0.99, 0.01]]])
    R = np.array(([[-0.45, -0.1],
                   [0.5,  0.5]]))
    return P, R, 0.9


def softmax(vals, temp=1.):
    """Batch softmax
    Args:
        vals (np.ndarray): S x A. Applied row-wise
        t (float, optional): Defaults to 1.. Temperature parameter
    Returns:
        np.ndarray: S x A
    """
    return np.exp((1./temp)*vals - logsumexp((1./temp)*vals, axis=1, keepdims=True))


def policy_evaluation(P, R, discount, policy):
    """ Policy Evaluation Solver
    We denote by 'A' the number of actions, 'S' for the number of
    states.
    Args:
      P (numpy.ndarray): Transition function as (A x S x S) tensor
      R (numpy.ndarray): Reward function as a (S x A) tensor
      discount (float): Scalar discount factor
      policies (numpy.ndarray): tensor of shape (S x A)
    Returns:
      tuple (vf, qf) where the first element is vector of length S and the second element contains
      the Q functions as matrix of shape (S x A).
    """
    nstates = P.shape[-1]
    ppi = np.einsum('ast,sa->st', P, policy)
    rpi = np.einsum('sa,sa->s', R, policy)
    vf = np.linalg.solve(np.eye(nstates) - discount*ppi, rpi)
    # qf = R + discount*np.einsum('ast,t->sa', P, vf)
    # return vf, qf
    return vf

def policy_performance(P, R, discount, initial_distribution, policy):
    """Expected discounted return from an initial distribution over states.
    Args:
        P (numpy.ndarray): Transition function as (A x S x S) array
        R (numpy.ndarray): Reward function as a (S x A) array
        discount (float): Scalar discount factor
        initial_distribution (numpy.ndarray): (S,) array
        policy (np.ndarray): (S x A) array
    Returns:
        float: Scalar performance
    """
    vf, _ = policy_evaluation(P, R, discount, policy)
    return initial_distribution @ vf


if __name__ == "__main__":
    mdp = dadashi_fig2d()
    nactions, nstates = mdp[0].shape[:2]

    temperature = 1.
    initial_distribution = np.ones(nstates)/nstates

    def objective(params):
        policy = softmax(params, temperature)
        return policy_performance(*mdp, initial_distribution, policy)

    objective = jit(objective)
    gradient = jit(grad(objective))
    params = np.zeros((nstates, nactions))
    for _ in range(500):
        params += 0.5*gradient(params)
        print(objective(params))
