import socket
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType


class LabyrinthGame(gym.Env):
    def __init__(
        self, num_rel_path=5, host: str = "127.0.0.1", port: int = 5000
    ) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, (64, 64, 1), np.uint8),
                "states": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
                "goal": gym.spaces.Box(
                    -np.inf, np.inf, (num_rel_path * 2,), np.float32
                ),
                "progress": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
            }
        )

        self.sock = socket.create_server((host, port))
        self.conn, _ = self.sock.accept()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return super().step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
        if self.conn is not None:
            self.conn.close()
