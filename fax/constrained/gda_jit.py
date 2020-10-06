import copy
import pickle
import jax
import jax.numpy as jnp
from jax import random, jit
from jax.lax import fori_loop
from jax.experimental import optimizers
from jax.experimental.stax import softmax
from jax.random import bernoulli
from jax.config import config

config.update("jax_enable_x64", True)



def get_new_key(key):
    mykey, subkey = random.split(key)
    key = subkey
    return key


def roll_out(last_state, last_action, p, model, key):
    key = get_new_key(key)
    s = bernoulli(key, p=p[last_action][last_state][0]).astype(int)
    key = get_new_key(key)
    a = bernoulli(key, p=model[s][0]).astype(int)
    return s, a, key


def sample_trajectory(policy, key, traj, init_state=False, my_s=0, init_action=False, my_a=0):
    true_transition = jnp.array([[[0.7, 0.3], [0.2, 0.8]],
                                 [[0.99, 0.01], [0.99, 0.01]]])
    initial_distribution = jnp.ones(2) / 2
    if init_state:
        s = my_s
    else:
        key = get_new_key(key)
        s = bernoulli(key, p=initial_distribution[0]).astype(int)
    if init_action:
        a = my_a
    else:
        key = get_new_key(key)
        a = bernoulli(key, p=policy[s][0]).astype(int)
    states = [s]
    actions = [a]
    # sample 2 times more for gae because of no absorbing state
    traj_len = len(traj)
    for _ in range(int(traj_len*2)):
        s, a, key = roll_out(s, a, true_transition, policy, key)
        states.append(s)
        actions.append(a)
    return states, actions, key


@jit
def get_log_policy(logits, state, action, policy_temperature = 1e-1):
    policy = softmax((1. / policy_temperature) * logits)
    return jnp.log(policy[state][action] + 1e-8)


def discounted_reward(rewards, t, traj_len, gamma=0.9):
    discounted = []
    for k in range(t, t + traj_len):
        discounted.append(gamma ** (k-t) * rewards[k])
    return jnp.array(discounted).sum()


@jit
def reinforce(model_logits, psi, states, actions):
    estimator = 0
    traj_len = int(len(states)/2)
    get_grad_log_policy = jit(jax.grad(get_log_policy, 0))
    for t in range(traj_len):
        state = states[t]
        action = actions[t]
        estimator += get_grad_log_policy(model_logits, state, action) * psi[t]
    return estimator / traj_len


def value_estimation(model_logits, key, traj_len, discriminator, policy_temperature=1e-1):
    policy = softmax((1. / policy_temperature) * model_logits)
    v0 = []
    v1 = []
    for _ in range(5):
        sample_length = int(traj_len/2)
        traj = jnp.ones((sample_length))
        states0, actions0, key = sample_trajectory(policy, key, traj, init_state=True, my_s=0)
        rewards0 = jnp.log(discriminator[(states0, actions0)] + 1e-8)
        states1, actions1, key = sample_trajectory(policy, key, traj, init_state=True, my_s=1)
        rewards1 = jnp.log(discriminator[(states1, actions1)] + 1e-8)
        v0.append(discounted_reward(rewards0, 0, traj_len))
        v1.append(discounted_reward(rewards1, 0, traj_len))
    v0 = jnp.array(v0).mean()
    v1 = jnp.array(v1).mean()
    return jnp.array([v0, v1]), key


def td1(values, rewards, states, t, gamma=0.9):
    next_state = states[t+1]
    cur_state = states[t]
    return rewards[t] + gamma * values[next_state] - values[cur_state]


def gae(values, states, actions, t, traj_len, discriminator, lmbda=0.9, gamma=0.9):
    estimator = 0
    deltas = []
    rewards = jnp.log(discriminator[(states, actions)] + 1e-8) # 100
    for k in range(t, t + traj_len):
        deltas.append(td1(values, rewards, states, k))
    for k in range(len(deltas)):
        estimator += ((lmbda * gamma) ** k) * deltas[k]
    return estimator


@jit
def get_policy_grad_naive(model_logits, states, actions, discriminator):
    traj_len = int(len(states)/2)
    discounted = []
    rewards = jnp.log(discriminator[(states, actions)] + 1e-8)

    for t in range(traj_len):
        discounted.append(discounted_reward(rewards, t, traj_len))
    return reinforce(model_logits, discounted, states, actions)


@jit
def get_policy_grad_gae(model_logits, states, actions, key, discriminator):
    advantages = []
    traj_len = int(len(states)/2)
    values, key = value_estimation(model_logits, key, traj_len, discriminator)
    for t in range(int(len(states)/2)):
        advantages.append(gae(values, states, actions, t, traj_len, discriminator))
    return reinforce(model_logits, advantages, states, actions), key


@jit
def get_policy_grad_td1(model_logits, states, actions, key, discriminator):
    tds = []
    traj_len = int(len(states)/2)
    values, key = value_estimation(model_logits, key, traj_len, discriminator)
    rewards = jnp.log(discriminator[(states, actions)] + 1e-8)
    for t in range(traj_len):
        tds.append(td1(values, rewards, states, t))
    return reinforce(model_logits, tds, states, actions), key



@jit
def discriminator_loss(discriminator_logits, states_model, actions_model, states_expert, actions_expert, traj,
                       discriminator_temperature=1e-2):
    traj_len = len(traj)
    states_model = states_model[:traj_len]
    actions_model = actions_model[:traj_len]
    states_expert = states_expert[:traj_len]
    actions_expert = actions_expert[:traj_len]

    discriminator = softmax((1. / discriminator_temperature) * discriminator_logits)
    negative = jnp.log(discriminator[(states_model, actions_model)] + 1e-10).sum()
    positive = jnp.log(1 - discriminator[(states_expert, actions_expert)] + 1e-10).sum()
    # gradient gradient_penalty
    return (negative + positive) / traj_len


@jit
def gen_loss(disc_logits, states, actions, traj, discriminator_temperature=1e-2):
    traj_len = len(traj)
    states = states[:traj_len]
    actions = actions[:traj_len]
    discriminator = softmax((1. / discriminator_temperature) * disc_logits)
    return jnp.log(discriminator[(states, actions)] + 1e-10).sum() / traj_len


print("=================start===================")

disc_logits = jnp.ones((2,2))
policy_logits = jnp.ones((2, 2))

disc_init, disc_update, disc_params = optimizers.adam(step_size=1e-3)
disc_state = disc_init(disc_logits)

policy_init, policy_update, policy_params = optimizers.adam(step_size=1e-3)
policy_state = policy_init(policy_logits)

policy_temperature = 1e-1
discriminator_temperature = 1e-2

key = random.PRNGKey(0)
length = 50
batch = 1
traj = jnp.ones((length))  # place hodler to pass traj len
policy_expert = jnp.array(([[0.4, 0.6],
                            [0.4,  0.6]]))

disc_losses = []
gen_losses = []
acc = []


for i in range(800):
    policy_model = softmax((1. / policy_temperature) * policy_logits)
    d = 1
    for j in range(d):
        disc_grad = jnp.zeros((2,2))
        for n in range(batch):
            states_model, actions_model, key = sample_trajectory(policy_model, key, traj)
            states_expert, actions_expert, key = sample_trajectory(policy_expert, key, traj)
            get_discriminator_grad = jax.grad(discriminator_loss, 0)
            grad = get_discriminator_grad(disc_logits, states_model, actions_model, states_expert, actions_expert, traj)
            disc_grad += grad

        disc_grad = disc_grad / batch
        disc_state = disc_update(i * d + j, disc_grad, disc_state)
        disc_logits = disc_params(disc_state)
        disc_loss = discriminator_loss(disc_logits, states_model, actions_model, states_expert, actions_expert, traj)
        disc_losses.append(disc_loss)

        print("disc grad", disc_grad)
        print("disc loss", disc_loss)
        print("disc logits", disc_logits)

    discriminator = softmax((1. / discriminator_temperature) * disc_logits)
    print("")

    p = 1
    for j in range(p):
        prev = policy_logits
        policy_grad = jnp.zeros((2, 2))
        for n in range(batch):
            states_model, actions_model, key = sample_trajectory(policy_model, key, traj)
            # grad, key = get_policy_grad_td1(policy_logits, states_model, actions_model, key, discriminator)
            grad, key = get_policy_grad_gae(policy_logits, states_model, actions_model, key, discriminator)
            # grad = get_policy_grad_naive(policy_logits, states_model, actions_model, discriminator)
            # grad = optimizers.clip_grads(grad, 0.5)
            policy_grad += grad

        policy_grad = policy_grad / batch
        policy_state = policy_update(i * p + j, policy_grad, policy_state)
        policy_logits = policy_params(policy_state)
        policy_model = softmax((1. / policy_temperature) * policy_logits)

        policy_loss = gen_loss(disc_logits, states_model, actions_model, traj)
        gen_losses.append(policy_loss)
        print("policy grad", policy_grad.flatten())
        print("policy loss", policy_loss)
        print("policy logits", policy_logits)

    print("policy",  policy_model)
    print("\n\n")

    with open('gen_loss.pkl', 'wb') as f:
        pickle.dump(gen_losses, f)
    with open('disc_loss.pkl', 'wb') as f:
        pickle.dump(disc_losses, f)

    if (i > 0) and jnp.max(jnp.abs(policy_logits - prev)) < 1e-6:
        print("converged")
        break

    prev = policy_logits
