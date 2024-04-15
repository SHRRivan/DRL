import gym

from algorithm.dqn import DQN
from algorithm.ddpg import DDPG
from algorithm.td3 import TD3
from algorithm.ppo import PPO
from algorithm.a2c import A2C

if __name__ == '__main__':
    # env = gym.make('BipedalWalkerHardcore-v3')
    env = gym.make('Pendulum-v1')
    model = A2C(env=env, policy='MLP', device='cpu')
    model.load('A2C', path=None)

    test_episodes = 10
    rewards = []
    for _ in range(test_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = model.predict(obs)
            next_state, reward, done, info = env.step(action)
            obs = next_state
            env.render()
            rewards.append(reward)
    env.close()
    del model
    print("Average reward is {}.".format(sum(rewards) / test_episodes))
    print("Total reward is {}.".format(sum(rewards)))
