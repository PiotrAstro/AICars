import numpy as np

class Abstract_Enviroment():
    def __init__(self):
        pass

    def react_to_action(self, action_index):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def is_alive(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_action_learning_state(self, action_probs):
        return np.random.choice(len(action_probs), p=action_probs)

    def get_action_production_state(self, action_probs):
        return np.argmax(action_probs)
