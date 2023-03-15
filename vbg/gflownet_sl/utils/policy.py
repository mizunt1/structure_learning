import numpy as np


def get_random_actions(masks, rng, weight=1):
    num_envs = masks.shape[0]

    # Reshape the masks, and add an entry for terminal action
    masks = masks.reshape(num_envs, -1)
    masks = np.column_stack((masks, weight * np.ones(num_envs)))  # Terminal action

    # Get uniform distribution over valid actions
    probas = masks / np.sum(masks, axis=1, keepdims=True)

    # Sample random actions with the above probabilities
    cumsum = np.cumsum(probas, axis=1)
    u = rng.random(size=(num_envs, 1))
    return np.sum(cumsum < u, axis=1)
