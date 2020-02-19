def create_env(env, **kwargs):
    """Expected names are: doom_battle, atari_montezuma, etc."""

    if env.startswith('doom_'):
        from seed_rl.envs.doom.doom_utils import make_doom_env
        env = make_doom_env(env, **kwargs)
    else:
        raise Exception('Unsupported env {0}'.format(env))

    return env
