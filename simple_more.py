import numpy as np
from core import World, Agent, Landmark
from scenario import BaseScenario


def expand_to_list(item, n):
    if type(item) == list:
        return item
    elif type(item) == str:
        if item.startswith("exp-"):
            e = float(item[4:])
            return [e ** (n - i - 1) for i in range(n)]
        elif item == "linear":
            return [n - i for i in range(n)]
    else:
        return [item] * n


class Scenario(BaseScenario):
    def make_world(self, num_targets=3, max_frames=1, reward_scales=1., size_scales=1., time_penalty=0.02,
                   game_end_after_touch=True, easy_mode=False):
        world = World()

        world.easy_mode = easy_mode
        if easy_mode:
            num_targets = 4
            # reward_scales = 1.
            # size_scales = 1.

        # add agents
        world.dim_p = 2
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_{}'.format(i)
            agent.collide = False
            agent.silent = True
            agent.size = 0.02
        # add landmarks
        world.num_targets = num_targets
        world.reward_scales = np.array(expand_to_list(reward_scales, num_targets))
        world.size_scales = np.array(expand_to_list(size_scales, num_targets))
        world.landmark_positions = []
        world.landmarks = [Landmark() for i in range(num_targets)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
            landmark.size *= world.size_scales[i]
        world.max_frames = max_frames
        world.time_penalty = time_penalty
        world.game_end_after_touch = game_end_after_touch
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.movable = True
            agent.color = np.array([0.6, 0.6, 0.6])
            agent.size = 0.02
        # random properties for landmarks
        rainbow_colors = [[255, 0, 0], [255, 127, 0], [255, 255, 0], [0, 255, 0],
                          [0, 0, 255], [75, 0, 130], [143, 0, 255]]
        color_names = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        rainbow_colors = np.array(rainbow_colors) / 255.0
        for i, landmark in enumerate(world.landmarks):
            # shade = i / (len(world.landmarks) - 1)
            # landmark.color = np.array([0.75, 0.75, 0.75]) - shade * 0.25
            # landmark.color = rainbow_colors[i]
            landmark.color = np.array([0.15, 0.15, 0.85])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.array([0., 0.])
            # if world.easy_mode:
            #     agent.state.p_pos = np.array([0., 0.])
            # else:
            #     agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            # agent.state.p_pos[0] = 0.
            # agent.state.p_pos[1] = 0.
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        world.landmark_positions = []
        dx = np.array([0, 1, 0, -1])
        dy = np.array([1, 0, -1, 0])
        for i, landmark in enumerate(world.landmarks):
            if world.easy_mode:
                landmark.state.p_pos = np.array([dx[i], dy[i]]) / 2
            else:
                landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        world.steps = 0
        world.touched = None

    def get_info(self, agent, world):
        return [np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)) for landmark in world.landmarks]

    def reward(self, agent, world, np_random):
        # dist2 = min([np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)) for landmark in world.landmarks])
        r = 0.

        world.turn_touched = False

        if agent.movable:
            rewards = 1. * world.reward_scales
            touched = False
            ind = list(range(world.num_targets))
            np_random.shuffle(ind)
            for i in ind:
                landmark = world.landmarks[i]
                if np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos))) <= agent.size + landmark.size + 0.01:
                    if world.touched is None or world.touched == i:
                        r += rewards[i]
                        touched = True
                        world.turn_touched = world.touched is None
                        world.touched = i
                        break
            if touched and world.game_end_after_touch:
                agent.movable = False
            if not touched:
                r -= world.time_penalty

        return r

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for i in range(len(world.landmarks)):
            entity_pos.append(world.landmarks[i].state.p_pos - agent.state.p_pos)
        # return np.concatenate([agent.state.p_vel] + entity_pos)
        # print(world.steps / world.max_frames)
        touched = np.zeros(world.num_targets)
        if world.touched is not None:
            touched[world.touched] = 1.
        return np.concatenate([np.array([world.steps / world.max_frames]), touched] +
                              [agent.state.p_vel] + entity_pos)

    def get_input_structure(self, agent, world):
        dim_p = world.dim_p
        input_structure = list()
        input_structure.append(("self", dim_p + 1))
        for _ in world.landmarks:
            input_structure.append(("landmarks", dim_p))
        return input_structure