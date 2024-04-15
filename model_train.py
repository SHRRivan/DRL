import gym

from algorithm.ppo import PPO

from common.monitor import Monitor

if __name__ == '__main__':
    env = gym.make('Pendulum-v1')

    env = Monitor(env)
    model = PPO(env, policy='MLP', tensorboard_log='walker', device='cpu')
    tb_ar_name = 'A2C'
    model.learn(total_timesteps=300_000, tb_log_name=tb_ar_name)
    model.save(algorithm_name=tb_ar_name)
    del model
