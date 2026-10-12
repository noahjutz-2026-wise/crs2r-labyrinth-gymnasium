import gymnasium as gym


class LabyrinthGame(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, 1)
