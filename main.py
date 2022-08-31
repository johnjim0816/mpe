from simple_env import SimpleEnv, make_env
from simple_more import Scenario
# from pettingzoo.utils.to_parallel import parallel_wrapper_fn


class raw_env(SimpleEnv):
    def __init__(self, max_frames, **kwargs):
        scenario = Scenario()
        world = scenario.make_world(max_frames=max_frames, **kwargs)
        super().__init__(scenario, world, max_frames)


# env = make_env(raw_env)
env = raw_env(1000)

# parallel_env = parallel_wrapper_fn(env)
state = env.reset()
for _ in range(1000):
    action = 0
    state = env.step(action)
    print(state)
    env.render()
    # observation = env.step(action)