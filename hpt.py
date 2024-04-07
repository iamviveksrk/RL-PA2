import numpy as np
from collections import namedtuple, deque
import datetime
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython.display import HTML
import optuna

########### DDQN HPT START #############

from agent_ddqn import DuelingQNetwork1, DuelingQNetwork2, Agent

def run_experiment(agent, n_episodes=200, max_t=1000):

    scores_window = deque(maxlen=100)

    episode_rewards = []
    begin_time = datetime.datetime.now()

    for i_episode in range(1, n_episodes+1):
        state = env.reset()[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            observation, reward, terminated, truncated, info = env.step(action)
            observation = np.array(observation)
            agent.step(state, action, reward, observation, (terminated or truncated))
            state = observation
            score += reward
            if (terminated or truncated):
                break

        scores_window.append(score)
        episode_rewards.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
          print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    return np.array(episode_rewards)

def objective(trial, q_network, state_shape, no_of_actions):

    GAMMA = 0.99  # discount factor (fixed)
    BUFFER_SIZE = int(1e5) # replay buffer size (larger the better)

    # Hyperparameters:

    BATCH_SIZE = 2 ** trial.suggest_int("exponent_batch_size", 3, 10)
    LR = trial.suggest_float("lr", 1e-5, 1, log=True)
    UPDATE_EVERY = trial.suggest_int("exponent_update_every", 15, 30)
    EPSILON = trial.suggest_float("epsilon", 1e-5, 0.1)
    MAX_GRAD_NORM = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    OPTIMIZER_NAME = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    agent = Agent(
                q_network=q_network,
                state_size=state_shape,
                action_size = no_of_actions,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                optimizer_name=OPTIMIZER_NAME,
                lr = LR,
                epsilon = EPSILON,
                gamma = GAMMA,
                update_every = UPDATE_EVERY,
                max_grad_norm = MAX_GRAD_NORM,
                seed = 0)

    episode_rewards = run_experiment(agent, n_episodes=100, max_t=env.spec.max_episode_steps)
    regret = -1 * episode_rewards

    return regret.mean()

types_ddqn = {'Type1': DuelingQNetwork1, 'Type2': DuelingQNetwork2}

for env_name in ['Acrobot-v1', 'CartPole-v1']:
  for network_type in ['Type1', 'Type2']:

    env = gym.make(env_name)

    state_shape = env.observation_space.shape[0]
    no_of_actions = env.action_space.n

    study = optuna.create_study(
        study_name=f'{env_name}-DDQN-{network_type}',
        direction='minimize',
        load_if_exists=True,
        storage=f'sqlite:///hpt_results.db',
    )

    study.optimize((lambda trial: objective(trial, types_ddqn[network_type], state_shape, no_of_actions)), n_trials=5)

############ DDQN HPT END ##############

######### REINFORCE HPT START ##########
from reinforce import run_reinforce
from actorCritic import run_actorCritic

def objective(trial, run_experiment, env, state_shape, no_of_actions):

    GAMMA = 0.99  # discount factor (fixed)

    # Hyperparameters:

    LR = trial.suggest_float("lr", 1e-5, 1, log=True)
    OPTIMIZER_NAME = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    episode_rewards = run_experiment(
                state_size=state_shape,
                action_size = no_of_actions,
                optimizer_name=OPTIMIZER_NAME,
                lr = LR,
                gamma = GAMMA,
                env = env,
                seed = 0)

    regret = -1 * episode_rewards

    return regret.mean()

types_reinforce = {'Type1': run_reinforce, 'Type2': run_actorCritic}

for env_name in ['Acrobot-v1', 'CartPole-v1']:
  for network_type in ['Type1', 'Type2']:

    env = gym.make(env_name)

    state_shape = env.observation_space.shape[0]
    no_of_actions = env.action_space.n

    study = optuna.create_study(
        study_name=f'{env_name}-REINFORCE-{network_type}',
        direction='minimize',
        load_if_exists=True,
        storage=f'sqlite:///hpt_results.db',
    )
    study.optimize((lambda trial: objective(trial, types_reinforce[network_type], env, state_shape, no_of_actions)), n_trials=5)

########## REINFORCE HPT END ###########
