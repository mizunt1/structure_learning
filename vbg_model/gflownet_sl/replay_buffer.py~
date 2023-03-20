import numpy as np
import math

from numpy.random import default_rng

from gflownet_sl.utils.multistep import get_nstep_mask


class ReplayBuffer:
    def __init__(self, capacity, num_variables, n_step=1, prioritized=False, **kwargs):
        self.capacity = capacity
        self.num_variables = num_variables
        self.n_step = n_step
        self.prioritized = prioritized

        nbytes = math.ceil((num_variables ** 2) / 8)
        dtype = np.dtype([
            ('adjacency', np.uint8, (nbytes,)),
            ('num_edges', np.int_, (1,)),
            ('actions', np.int_, (1,)),
            ('is_exploration', np.bool_, (1,)),
            ('rewards', np.float_, (1,)),
            ('scores', np.float_, (1,)),
            ('mask', np.uint8, (nbytes,)),
            ('next_adjacency', np.uint8, (nbytes,)),
            ('next_mask', np.uint8, (nbytes,))
        ])
        self._replay = np.zeros((capacity,), dtype=dtype)
        self._index = 0
        self._is_full = False
        self._prev = np.full((capacity,), -1, dtype=np.int_)
        self._zeros = np.zeros((), dtype=dtype)

    def add(self, observations, actions, is_exploration, next_observations, rewards, dones, prev_indices=None):
        indices = np.full((dones.shape[0],), -1, dtype=np.int_)
        if np.all(dones):
            return indices

        num_samples = np.sum(~dones)
        add_idx = np.arange(self._index, self._index + num_samples) % self.capacity
        self._is_full |= (self._index + num_samples >= self.capacity)
        self._index = (self._index + num_samples) % self.capacity
        indices[~dones] = add_idx

        data = {
            'adjacency': self.encode(observations['adjacency'][~dones]),
            'num_edges': observations['num_edges'][~dones],
            'actions': actions[~dones],
            'is_exploration': is_exploration[~dones],
            'rewards': rewards[~dones],
            'scores': observations['score'][~dones],
            'mask': self.encode(observations['mask'][~dones]),
            'next_adjacency': self.encode(next_observations['adjacency'][~dones]),
            'next_mask': self.encode(next_observations['mask'][~dones])
        }

        for name in data:
            shape = self._replay.dtype[name].shape
            self._replay[name][add_idx] = np.asarray(data[name].reshape(-1, *shape))
        
        if prev_indices is not None:
            self._prev[add_idx] = prev_indices[~dones]

        return indices

    def sample(self, batch_size, rng=default_rng()):
        # Get all the indices for the n-steps
        indices = np.full((self.n_step, batch_size), -1, dtype=np.int_)
        indices[0] = rng.choice(len(self), size=batch_size, replace=False)
        for i in range(1, self.n_step):
            indices[i] = np.where(
                indices[i - 1] >= 0,
                self._prev[indices[i - 1]], -1
            )

        lengths = np.sum(indices >= 0, axis=0)
        samples = self._replay[indices]
        samples[indices < 0] = self._zeros  # Mask out invalid samples

        # Convert structured array into dictionary
        return ({
            'adjacency': self.decode(samples['adjacency']),
            'num_edges': samples['num_edges'],
            'actions': samples['actions'],
            'rewards': samples['rewards'],
            'mask': self.decode(samples['mask']),
            'next_adjacency': self.decode(samples['next_adjacency']),
            'next_mask': self.decode(samples['next_mask'])
        }, get_nstep_mask(lengths, self.n_step))

    def __len__(self):
        return self.capacity if self._is_full else self._index

    @property
    def transitions(self):
        return self._replay[:len(self)]

    def save(self, filename, *, safe=True):
        data = {
            'version': 2,
            'replay': self.transitions,
            'index': self._index,
            'is_full': self._is_full,
            'prev': self._prev
        }
        np.savez_compressed(filename, **data)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = np.load(f)
            self._index = data['index']
            self._is_full = data['is_full']
            self._prev = data['prev']
            self._replay[:len(self)] = data['replay']
        return self

    def encode(self, decoded):
        encoded = decoded.reshape(-1, self.num_variables ** 2)
        return np.packbits(encoded, axis=1)

    def decode(self, encoded, dtype=np.float32):
        decoded = np.unpackbits(encoded, axis=-1, count=self.num_variables ** 2)
        decoded = decoded.reshape(*encoded.shape[:-1], self.num_variables, self.num_variables)
        return decoded.astype(dtype)

    @property
    def dummy_adjacency(self):
        return np.zeros((self.num_variables, self.num_variables), dtype=np.float32)

    def update_priorities(self, samples, priorities):
        pass


if __name__ == '__main__':
    replay = ReplayBuffer(7, 5)
    
    observations = {
        'adjacency': np.random.binomial(1, 0.5, size=(3, 5, 5)),
        'num_edges': np.random.randint(0, 11, size=(3,)),
        'mask': np.random.binomial(1, 0.5, size=(3, 5, 5)),
        'score': np.random.randn(3)
    }
    actions = np.random.randint(0, 25, size=(3,))
    is_exploration = np.random.binomial(1, 0.5, size=(3,))
    rewards = np.random.randn(3)
    dones = np.array([False, False, False])

    indices = None
    for _ in range(3):
        rewards = np.random.randn(3)
        indices = replay.add(observations, actions, is_exploration, observations, rewards, dones, prev_indices=indices)
    
    samples, _ = replay.sample(batch_size=3, n_step=2)
    print(replay._replay)
