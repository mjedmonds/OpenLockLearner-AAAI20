from collections import namedtuple
import random

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple(
    "Transition", ("state", "action", "mask", "next_state", "reward")
)


class Memory(object):
    def __init__(self, limit=0):
        self.memory = []
        self.limit = limit

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))
        if self.limit != 0 and len(self.memory) > self.limit:
            self.remove_last()

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        if isinstance(new_memory, Memory):
            self.memory += new_memory.memory
        else:
            self.memory += new_memory
        while self.limit != 0 and len(self.memory) > self.limit:
            self.remove_last()

    def remove_last(self):
        try:
            self.memory.pop(0)
        except:
            pass

    def clear(self):
        del self.memory
        self.memory = []

    def __len__(self):
        return len(self.memory)


class PrioritizedMemory(Memory):
    def __init__(self, limit=0):
        super(PrioritizedMemory, self).__init__(limit=limit)

    def sample(self, batch_size=None, rand_scale=5):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(
                self.memory[: batch_size * rand_scale], batch_size
            )
            return Transition(*zip(*random_batch))

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))
        self.sort()
        if self.limit != 0 and len(self.memory) > self.limit:
            self.remove_last()

    def append(self, new_memory):
        if isinstance(new_memory, Memory):
            self.memory += new_memory.memory
        else:
            self.memory += new_memory
        self.sort()
        while self.limit != 0 and len(self.memory) > self.limit:
            self.remove_last()

    def sort(self):
        self.memory.sort(key=lambda x: x.reward, reverse=True)

    def remove_last(self):
        try:
            self.memory.pop()
        except:
            pass

    def clear(self):
        del self.memory
        self.memory = []
