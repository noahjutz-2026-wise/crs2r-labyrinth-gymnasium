import pickle
import socket
import struct
import threading
import time

import gymnasium as gym

import crs2r_labyrinth_gymnasium
from crs2r_labyrinth_gymnasium.labyrinth_game import LabyrinthGame


def test_make_env():
    threading.Thread(
        target=lambda: (time.sleep(0.05), socket.create_connection(("127.0.0.1", 5000)).close())
    ).start()
    env = gym.make("crs2r/LabyrinthGame-v0")
    assert env is not None
    env.close()


def test_socket_exchange():
    def send(sock, obj):
        payload = pickle.dumps(obj, protocol=4)
        sock.sendall(struct.pack("!I", len(payload)) + payload)

    def recv(sock):
        size = struct.unpack("!I", sock.recv(4))[0]
        return pickle.loads(sock.recv(size))

    def client():
        time.sleep(0.05)
        s = socket.create_connection(("127.0.0.1", 5000))

        # Reset handshake
        assert recv(s) == (42, None)
        send(s, ("obs", {}))

        # Step handshake
        assert recv(s) == "action"
        send(s, ("obs", 1.0, False, False, {}))

        s.close()

    thread = threading.Thread(target=client)
    thread.start()

    env = LabyrinthGame()
    obs, info = env.reset(seed=42)
    assert obs == "obs"

    obs, reward, terminated, truncated, info = env.step("action")
    assert reward == 1.0

    env.close()
    thread.join()
