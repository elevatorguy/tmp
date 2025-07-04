from pdb import set_trace as T

import gym
import shimmy
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments


EXTRA_OBS_KEYS = [
    'tty_chars',
    'tty_colors',
    'tty_cursor',
]

ALIASES = {
    'minihack': 'MiniHack-River-v0',
}

def env_creator(name='minihack'):
    return functools.partial(make, name)

def make(name, buf=None, seed=0):
    '''NetHack binding creation function'''
    if name in ALIASES:
        name = ALIASES[name]

    import minihack
    pufferlib.environments.try_import('minihack')
    obs_key = minihack.base.MH_DEFAULT_OBS_KEYS + EXTRA_OBS_KEYS
    env = gym.make(name, observation_keys=obs_key)
    env = shimmy.GymV21CompatibilityV0(env=env)
    env = MinihackWrapper(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

class MinihackWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.close = self.env.close
        self.close = self.env.close
        self.render_mode = 'ansi'

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.obs = obs
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.obs = obs
        return obs, reward, done, truncated, info

    def render(self):
        import nle
        chars = nle.nethack.tty_render(
            self.obs['tty_chars'], self.obs['tty_colors'], self.obs['tty_cursor'])
        return chars

