import gymnasium as gym

import crs2r_labyrinth_gymnasium


def test_make_env():
    env = gym.make("crs2r/LabyrinthGame-v0")
    assert env is not None
    env.close()
