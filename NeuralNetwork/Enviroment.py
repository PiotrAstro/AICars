

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
