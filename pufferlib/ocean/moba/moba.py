from pdb import set_trace as T
import numpy as np
import os

import pettingzoo
import gymnasium

import pufferlib
from pufferlib.ocean.moba import binding

MAP_OBS_N = 11*11*4
PLAYER_OBS_N = 26

class Moba(pufferlib.PufferEnv):
    def __init__(self, num_envs=4, vision_range=5, agent_speed=1.0,
            discretize=True, reward_death=-1.0, reward_xp=0.006,
            reward_distance=0.05, reward_tower=3.0, report_interval=32,
            script_opponents=True, render_mode='human', buf=None, seed=0):

        self.report_interval = report_interval
        self.render_mode = render_mode
        self.num_agents = 5*num_envs if script_opponents else 10*num_envs

        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(MAP_OBS_N + PLAYER_OBS_N,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 7, 3, 2, 2, 2])

        super().__init__(buf=buf)

        c_envs = []
        players = 5 if script_opponents else 10
        self.c_state = binding.shared()
        for i in range(num_envs):
            env_id = binding.env_init(
                self.observations[i*players:(i+1)*players],
                self.actions[i*players:(i+1)*players],
                self.rewards[i*players:(i+1)*players],
                self.terminals[i*players:(i+1)*players],
                self.truncations[i*players:(i+1)*players],
                i + seed*num_envs,
                vision_range=vision_range,
                agent_speed=agent_speed,
                discretize=discretize,
                reward_death=reward_death,
                reward_xp=reward_xp,
                reward_distance=reward_distance,
                reward_tower=reward_tower,
                script_opponents=script_opponents,
                state=self.c_state,
            )
            c_envs.append(env_id)

        self.c_envs = binding.vectorize(*c_envs)
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.actions[:, 0] = 100*(self.actions[:, 0] - 3)
        self.actions[:, 1] = 100*(self.actions[:, 1] - 3)
        binding.vec_step(self.c_envs)

        infos = []
        self.tick += 1
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                infos.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, infos)

    def render(self):
        for frame in range(12):
            binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=20, atn_cache=1024, num_envs=400):
    tick = 0

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: %f', 10*num_envs*tick / (time.time() - start))

if __name__ == '__main__':
    # Run with c profile
    from cProfile import run
    num_envs = 400
    env = Moba(num_envs=num_envs, report_interval=10000000)
    env.reset()
    actions = np.random.randint(0, env.single_action_space.nvec, (1024, 10*num_envs, 6))
    test_performance(20, 1024, num_envs)
    exit(0)

    run('test_performance(20)', 'stats.profile')
    import pstats
    from pstats import SortKey
    p = pstats.Stats('stats.profile')
    p.sort_stats(SortKey.TIME).print_stats(25)
    exit(0)

    #test_performance(10)
