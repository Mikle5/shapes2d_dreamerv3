def main():

    import warnings
    import dreamerv3
    import gym
    import envs.shapes2d
    import env_rescale
    from dreamerv3 import embodied
    import crafter
    from embodied.envs import from_gym
    from gym.spaces.box import Box
    import os
    import sys
    import tensorflow as tf
    import jax
    import numpy as np
    import env_rescale
    import obs_resize
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')
    os.system("nvidia-smi")
  # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['medium'])
    config = config.update({
    'logdir': '~/dreamer_logs/Pushing7x7-v0/dreamerv3/run1',
    'run.train_ratio': 64,
    'run.log_every': 30,  # Seconds
    'batch_size': 16,
    'jax.prealloc': False,
    'encoder.mlp_keys': '$^',
    'decoder.mlp_keys': '$^',
    'encoder.cnn_keys': 'image',
    'decoder.cnn_keys': 'image',
    'jax.platform': 'cpu',
    })
    config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
#        embodied.logger.WandBOutput(logdir.name, config),
#        embodied.logger.MLFlowOutput(logdir.name),
  ])
#    env = gym.make('Navigation5x5-v0')
#    env.reset()
#    env = env_rescale.atari_env_eval(env)
#    env = crafter.Env()  # Replace this with your Gym env.
    env = from_gym.FromGym('Pushing7x7-v0', obs_key='image')  # Or obs_key='vector'.
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay')
    args = embodied.Config(config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    embodied.run.train(agent, env, replay, logger, args)
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
    main()

