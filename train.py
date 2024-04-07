import numpy as np
from collections import namedtuple, deque
import datetime
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import optuna
from agent_ddqn import DuelingQNetwork1, DuelingQNetwork2, Agent
from reinforce import run_reinforce
from actorCritic import run_actorCritic

plt.rcParams['font.family'] = 'serif'
legend_handles = lambda color, label: mlines.Line2D([], [], color=color, marker='s', linestyle='None', markersize=10, label=label, markerfacecolor='none')

studies = [f'{env_name}-{algo}-{network_type}' for env_name in ['Acrobot-v1', 'CartPole-v1'] for network_type in ['Type1', 'Type2'] for algo in ['DDQN', 'REINFORCE']]

# df = pd.DataFrame()
# df['studies'] = studies

# print(df)

types_ddqn = {'Type1': DuelingQNetwork1, 'Type2': DuelingQNetwork2}
types_reinforce = {'Type1': run_reinforce, 'Type2': run_actorCritic}

for algo in ['REINFORCE', 'DDQN']:
    for env_name in ['CartPole-v1', 'Acrobot-v1']:

        env = gym.make(env_name)
        state_shape = env.observation_space.shape[0]
        no_of_actions = env.action_space.n

        runs = []
        GAMMA = 0.99

        for network_type in ['Type1', 'Type2']:
            study = optuna.load_study(study_name=f'{env_name}-{algo}-{network_type}', storage="sqlite:///hpt_results.db")
            hpt_params = study.best_params

            run_i = []

            if algo == 'REINFORCE':
                n_episodes = 600
                LR = hpt_params["lr"]
                OPTIMIZER_NAME = hpt_params["optimizer"]

                for seed in range(5):

                  episode_rewards = types_reinforce[network_type](
                              state_size=state_shape,
                              action_size = no_of_actions,
                              optimizer_name=OPTIMIZER_NAME,
                              lr = LR,
                              gamma = GAMMA,
                              env = env,
                              n_episodes=n_episodes,
                              seed = seed)
                  run_i.append(episode_rewards)

            else:
                n_episodes = 300
                BUFFER_SIZE = int(1e5)

                # Hyperparameters from best_params

                BATCH_SIZE = 2 ** hpt_params["exponent_batch_size"]
                LR = hpt_params["lr"]
                UPDATE_EVERY = hpt_params["exponent_update_every"]
                EPSILON = hpt_params["epsilon"]
                MAX_GRAD_NORM = hpt_params["max_grad_norm"]
                OPTIMIZER_NAME = hpt_params["optimizer"]

            

                for seed in range(5):
                    agent = Agent(
                            q_network=types_ddqn[network_type],
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
                            seed = seed)

                    scores_window = deque(maxlen=100)
                    episode_rewards = []

                    for i_episode in range(1, n_episodes+1):
                        state = env.reset()[0]
                        score = 0
                        for t in range(env.spec.max_episode_steps):
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
                    
                    run_i.append(episode_rewards)
            
            runs.append(np.array(run_i))

        plt.figure(figsize=(10, 6), dpi=120)
        plt.box(False)

        plt.grid(axis='y', linestyle='-', linewidth=1)
        plt.xlabel('Episode Number', alpha=0.5)
        plt.ylabel('Episodic Return', alpha=0.5)

        plt.title(f'{algo}\n({env_name})', ha='left', x=0, alpha=0.5)
        plt.plot(np.arange(1, n_episodes+1), runs[0].mean(axis=0), color='red')
        plt.fill_between(np.arange(1, n_episodes+1), runs[0].mean(axis=0)-runs[0].std(axis=0), runs[0].mean(axis=0)+runs[0].std(axis=0), alpha=0.1, color='red')

        plt.plot(np.arange(1, n_episodes+1), runs[1].mean(axis=0), color='blue')
        plt.fill_between(np.arange(1, n_episodes+1), runs[1].mean(axis=0)-runs[1].std(axis=0), runs[1].mean(axis=0)+runs[1].std(axis=0), alpha=0.1, color='blue')
        plt.legend(loc=9, ncols=2, bbox_to_anchor=(0.5, 1.1), frameon = False, labelcolor='linecolor', handles=[legend_handles('red', 'Type-1'), legend_handles('blue', 'Type-2')])

        plt.savefig(f'plots/{algo}\n({env_name}).png', bbox_inches='tight',transparent=True, pad_inches=0)
        # plt.show()