from gymnasium.envs.registration import register

register(
    id="crs2r/LabyrinthGame-v0",
    entry_point="crs2r_labyrinth_gymnasium:LabyrinthGame",
    max_episode_steps=300,
)
