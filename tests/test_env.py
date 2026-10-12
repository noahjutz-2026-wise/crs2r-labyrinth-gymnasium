import threading

import gymnasium as gym
import zmq

import crs2r_labyrinth_gymnasium
from crs2r_labyrinth_gymnasium.labyrinth_game import LabyrinthGame


def test_make_env():
    env = gym.make("crs2r/LabyrinthGame-v0")
    assert env is not None
    env.close()


def test_socket_exchange():
    def client():
        ctx = zmq.Context()
        s = ctx.socket(zmq.REP)
        s.connect("tcp://127.0.0.1:5000")

        # Reset handshake
        assert s.recv_pyobj() == (42, None)
        s.send_pyobj(("obs", {}))

        # Step handshake
        assert s.recv_pyobj() == "action"
        s.send_pyobj(("obs", 1.0, False, False, {}))

        s.close()
        ctx.term()

    thread = threading.Thread(target=client)
    thread.start()

    env = LabyrinthGame()
    obs, info = env.reset(seed=42)
    assert obs == "obs"

    obs, reward, terminated, truncated, info = env.step("action")
    assert reward == 1.0

    env.close()
    thread.join()
