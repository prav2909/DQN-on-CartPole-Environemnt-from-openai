import gym
from main import DqnAgent
from replay_buffer import ReplayBuffer
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
def evaluate_training_result(env, agent):

    total_reward = 0.0
    episodes_to_play = 10

    for i in range(episodes_to_play):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    average_reward = total_reward / episodes_to_play
    return average_reward

def collect_gameplay_experiences(env, agent, buffer):
    """
    collect gameplay experiences
    :param env: the game environemnt
    :param agent: the DQN Agent
    :param buffer: the replay buffer
    :return: None
    """

    state = env.reset()
    done = False
    while not done:
        action = agent.collect_policy(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -1.0
        buffer.store_gameplay_experience(state, next_state,
                                         reward, action, done)
        state = next_state

def train_model(max_episodes=5000):
    """
    Train the model
    :param max_episodes: max number episoedes to train model
    :return: None
    """
    agent = DqnAgent()
    buffer = ReplayBuffer()
    env = gym.make('CartPole-v0')
    for _ in range(100):
        collect_gameplay_experiences(env, agent, buffer)
    for episode_cnt in range(max_episodes):
        collect_gameplay_experiences(env, agent, buffer)
        gameplay_experience_batch = buffer.sample_gameplay_batch()
        loss = agent.train(gameplay_experience_batch)
        avg_reward = evaluate_training_result(env, agent)
        env.render()
        print('Episode {0}/{1} and so far the performance is {2} and loss is {3}'.format(episode_cnt, max_episodes, avg_reward, loss[0]))

        if episode_cnt % 20 == 0:
            agent.update_target_network()
    env.close()


train_model()