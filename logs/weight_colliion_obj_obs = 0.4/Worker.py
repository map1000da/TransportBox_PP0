
from myenv.env import MyEnv
import multiprocessing
import multiprocessing.connection

def worker_process(remote: multiprocessing.connection.Connection, parallel_number: int):

    # create game
    game = MyEnv()
    # wait for instructions from the connection and execute them
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(game.step(data))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError

class Worker:
    """
    Creates a new worker and runs it in a separate process.
    """

    def __init__(self, parallel_number):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, parallel_number))
        self.process.start()
