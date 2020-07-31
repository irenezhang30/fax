import copy

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import optimizers
from jax.experimental.stax import softmax
from jax.random import bernoulli

true_transition = jnp.array([[[0.4, 0.6], [0.3, 0.7]],
                             [[0.8, 0.2], [0.8, 0.2]]])
temperature = 1e-2
true_discount = 0.9
traj_len = 100
initial_distribution = jnp.ones(2) / 2
# model policy should converge to this value
policy_expert = jnp.array(([[0.4, 0.6],
                            [0.4,  0.6]]))
key = random.PRNGKey(0)


def get_new_key():
    global key
    mykey, subkey = random.split(key)
    key = subkey


def roll_out(last_state, last_action, p, model):
    global key
    get_new_key()
    s = bernoulli(key, p=p[last_action][last_state][0]).astype(int)
    get_new_key()
    a = bernoulli(key, p=model[s][0]).astype(int)
    return (s, a)


def sample_trajectory(policy):
    global key
    get_new_key()
    s = bernoulli(key, p=initial_distribution[0]).astype(int)
    get_new_key()
    a = bernoulli(key, p=policy[s][0]).astype(int)
    traj = []
    traj.append((s, a))
    for i in range(traj_len - 1):
        s, a = roll_out(s, a, true_transition, policy)
        traj.append((s, a))
    return jnp.array(copy.deepcopy(traj))


def discriminator_loss(discriminator_logits, traj_model, traj_expert):
    discriminator = softmax((1. / temperature) * discriminator_logits)
    loss = 0
    for i in range(traj_len):
        s_model, a_model = traj_model[i]
        s_expert, a_expert = traj_expert[i]
        loss += jnp.log(discriminator[s_model][a_model]) + jnp.log(1 - discriminator[s_expert][a_expert])
    return loss / traj_len


def get_log_policy(model_logits, s, a):
    policy_model = softmax((1. / temperature) * model_logits)
    return jnp.log(policy_model[s][a])


get_grad_log_policy = jax.grad(get_log_policy)


def discounted_reward(rewards, t, gamma=0.9):
    discounted = [gamma ** (i - t) * rewards[i] for i in range(t, len(rewards))]
    G = jnp.array(discounted).sum()
    return G


def sample_rewards(policy, discriminator, initialization=False, initial_state=0):
    global key
    get_new_key()
    s = bernoulli(key, p=initial_distribution[0]).astype(int)
    if initialization:
        s = initial_state
    get_new_key()
    a = bernoulli(key, p=policy[s][0]).astype(int)

    traj = []
    traj.append((s, a))
    returns = []
    returns.append(discriminator[s][a])

    for i in range(traj_len - 1):
        s, a = roll_out(s, a, true_transition, policy)
        traj.append((s, a))
        returns.append(discriminator[s][a])

    return jnp.array(copy.deepcopy(returns)), jnp.array(copy.deepcopy(traj))



def value_estimation_first_visit(discriminator_logits, model_logits):
    discriminator = softmax((1. / temperature) * discriminator_logits)
    policy = softmax((1. / temperature) * model_logits)
    v0 = []
    v1 = []
    for i in range(10):
        rewards0, _ = sample_rewards(policy, discriminator, initialization=True, initial_state=0)
        rewards1, _ = sample_rewards(policy, discriminator, initialization=True, initial_state=1)
        v0.append(discounted_reward(rewards0, 0))
        v1.append(discounted_reward(rewards1, 0))
    v0 = jnp.array(v0).mean()
    v1 = jnp.array(v1).mean()
    return jnp.array([v0, v1])


def value_loss(value, approx):
    return jnp.linalg.norm(value - approx)

# from value function estimation
def value_approximiation(discriminator_logits, model_logits, batch=5, threshold=1e-2):
    discriminator = softmax((1. / temperature) * discriminator_logits)
    policy_model = softmax((1. / temperature) * model_logits)

    v0_rewards, _ = sample_rewards(policy_model, 0, discriminator)
    v1_rewards, _ = sample_rewards(policy_model, 1, discriminator)
    value = jnp.array([discounted_reward(v0_rewards, 0), discounted_reward(v1_rewards, 0)])

    opt_init3, opt_update3, get_params3 = optimizers.adam(step_size=0.01)
    opt_state3 = opt_init3(value)

    prev = value
    for i in range(30):
        rewards, _ = sample_rewards(policy_model, 0, discriminator)
        v0 = discounted_reward(rewards, 0)

        rewards, _= sample_rewards(policy_model, 1, discriminator)
        v1 = discounted_reward(rewards, 0)

        print ("check", v0, v1)
        grad_loss = jax.grad(value_loss, (0))(value, jnp.array([v0, v1]))

        opt_state3 = opt_update3(i, grad_loss, opt_state3)
        value = get_params(opt_state3)
        print ("value: ", value.flatten(), "prev: ", prev.flatten())

        if i > 0 and abs(jnp.max(value - prev)) <= threshold:
            print (value - prev)
            print ("converged in ", i)
            return value
        prev = copy.deepcopy(get_params(opt_state3))


    print ("************not converged")
    return value


def reinforce(discriminator_logits, model_logits, rewards, traj_model):
    estimator = 0
    for i in range(traj_len):
        s_model, a_model = traj_model[i]
        estimator += get_grad_log_policy(model_logits, s_model, a_model) * rewards[i]
    return estimator


def gae(values, rewards, traj_model, t, lmbda=0.9, gamma=0.9):
    rewards = rewards[t:]
    traj_model = traj_model[t:]
    estimator = 0

    deltas = []
    for i in range(len(rewards)):
        next_state = traj_model[1][0]
        cur_state = traj_model[0][0]
        deltas.append(rewards[i] + gamma * values[next_state] * - values[cur_state])

    for i in range(len(rewards)):
        estimator += lmbda ** i * deltas[i]

    return estimator


# reinforce
def get_policy_grad_naive(discriminator_logits, model_logits, traj_model):
    discriminator = softmax((1. / temperature) * discriminator_logits)
    estimator = 0

    gen_losses = []
    rewards = []

    for i in range(traj_len):
        s_model, a_model = traj_model[i]
        gen_losses.append((jnp.log(discriminator[s_model][a_model])))

    for i in range(traj_len):
        rewards.append(discounted_reward(gen_losses, i, gamma=0.9))

    return reinforce(discriminator_logits, model_logits, rewards, traj_model)

# reinforce with gae
def get_policy_grad_gae(discriminator_logits, model_logits, traj_model):
    discriminator = softmax((1. / temperature) * discriminator_logits)

    rewards = []
    values = []
    returns = []
    advantages = []

    for i in range(traj_len):
        s_model, a_model = traj_model[i]
        rewards.append(jnp.log(discriminator[s_model][a_model]))

    values = value_estimation_first_visit(discriminator_logits, model_logits)
    for t in range(traj_len):
        advantages.append(gae(values, rewards, traj_model, t))
    return reinforce(discriminator_logits, model_logits, advantages, traj_model)


def update_discriminator(i, discriminator_logits, traj_model, traj_expert, opt_state, opt_update, get_params):
    discriminator_grad =jax.grad(discriminator_loss)(discriminator_logits, traj_model, traj_expert)
    opt_state = opt_update(i, discriminator_grad, opt_state)
    discriminator_logits = get_params(opt_state)
    return opt_state, discriminator_logits


def update_policy(t, discriminator_logits, model_logits, traj_model, opt_state, opt_update, get_params):
    policy_grad = get_policy_grad_gae(discriminator_logits, model_logits, traj_model)
    opt_state = opt_update(i, policy_grad, opt_state)
    model_logits = get_params(opt_state)
    return opt_state, model_logits


# initialization
discriminator_logits = jnp.ones((2,2))
model_logits = jnp.ones((2,2))

opt_init, opt_update, get_params = optimizers.adam(step_size=0.001)
opt_state = opt_init(discriminator_logits)

opt_init2, opt_update2, get_params2 = optimizers.adam(step_size=0.001)
opt_state2 = opt_init2(model_logits)

for i in range(50):
    policy_model = softmax((1. / temperature) * model_logits)
    traj_model = sample_trajectory(policy_model)
    traj_expert = sample_trajectory(policy_expert)
    opt_state, discriminator_logits = update_discriminator(i, discriminator_logits, traj_model, traj_expert, opt_state,
                                                           opt_update, get_params)
    opt_state2, model_logits = update_policy(i, discriminator_logits, model_logits, traj_model, opt_state2, opt_update2,
                                             get_params2)

    print ("discriminator_logits: \n", discriminator_logits)
    print ("model_logits: \n", model_logits)
    print ("policy:  \n", softmax((1. / temperature) * model_logits))
    print ("")


