import math

from NeuralNetwork.Enviroment import Abstract_Enviroment


class Car_Enviroment(Abstract_Enviroment):
    def __init__(self,
                 game_map,
                 WIDTH=800, HEIGHT=800,
                 CAR_SIZE_X=46, CAR_SIZE_Y=27,
                 BORDER_COLOR=(255, 255, 255),
                 STARTX=400, STARTY=730,
                 START_SPEED=2,
                 MAX_SPEED=10, MIN_SPEED=1,
                 SPEED_CHANGE=1,
                 START_DIRECTION=math.radians(0),
                 DIRECTION_CHANGE=math.radians(1),
                 DIRECTION_CHECK_NUMBER=9,
                 DIRECTION_CHECK_CHANGE=math.radians(45)):
        super().__init__()

        self.game_map = game_map
        self.WIDTH, self.HEIGHT = WIDTH, HEIGHT
        self.CAR_SIZE_X, self.CAR_SIZE_Y = CAR_SIZE_X, CAR_SIZE_Y
        self.BORDER_COLOR = BORDER_COLOR
        self.STARTX, self.STARTY = STARTX, STARTY
        self.START_SPEED = START_SPEED
        self.MAX_SPEED, self.MIN_SPEED = MAX_SPEED, MIN_SPEED
        self.SPEED_CHANGE = SPEED_CHANGE
        self.START_DIRECTION = START_DIRECTION
        self.DIRECTION_CHANGE = DIRECTION_CHANGE
        self.DIRECTION_CHECK_NUMBER = DIRECTION_CHECK_NUMBER
        self.DIRECTION_CHECK_CHANGE = DIRECTION_CHECK_CHANGE
        self.reset()

    def step(self):
        if self.alive:
            if self.next_state is None:
                self.next_state = self.create_state(self.game_map)
            self.states.append(self.next_state)
            action = self.PPO.get_action(self.next_state)
            self.actions.append(action)

            self.use_action(action)
            self.update()

            self.last_reward = self.get_reward(self.game_map)

    def use_action(self, action):
        if action == 0:
            pass
        elif action == 1:
            self.rotate("LEFT")
        elif action == 2:
            self.rotate("RIGHT")
        # elif action == 3:
        #     self.speed_change("UP")
        # elif action == 4:
        #     self.speed_change("DOWN")
        # elif action == 5:
        #     self.rotate("LEFT")
        #     self.speed_change("UP")
        # elif action == 6:
        #     self.rotate("LEFT")
        #     self.speed_change("DOWN")
        # elif action == 7:
        #     self.rotate("RIGHT")
        #     self.speed_change("UP")
        # elif action == 8:
        #     self.rotate("RIGHT")
        #     self.speed_change("DOWN")

    def get_reward_car(self, game_map):
        if self.check_collision(game_map):
            self.alive = False
            return -10
        else:
            self.next_state = self.create_state(game_map)
            minimum_distance = 100
            for i in range(self.DIRECTION_CHECK_NUMBER):
                if self.next_state[0][i + 1] < minimum_distance:
                    minimum_distance = self.next_state[0][i + 1]
            returnValue = minimum_distance # + (self.next_state[0][0]) / 10
            return returnValue

    def update(self):
        self.position[0] += math.cos(self.direction) * self.speed
        self.position[1] -= math.sin(self.direction) * self.speed  # y-axis is inverted

    def check_collision(self, game_map):
        cos_theta = math.cos(self.direction)
        sin_theta = math.sin(self.direction)

        x_half = self.CAR_SIZE_X / 2
        y_half = self.CAR_SIZE_Y / 2

        x_offset1 = x_half * cos_theta - y_half * sin_theta
        y_offset1 = x_half * sin_theta + y_half * cos_theta
        x_offset2 = -x_half * cos_theta - y_half * sin_theta
        y_offset2 = -x_half * sin_theta + y_half * cos_theta

        # y-axis is inverted
        corners = [
            (self.position[0] + x_offset1, self.position[1] - y_offset1),  # Top-right corner
            (self.position[0] - x_offset1, self.position[1] + y_offset1),  # Top-left corner
            (self.position[0] - x_offset2, self.position[1] + y_offset2),  # Bottom-left corner
            (self.position[0] + x_offset2, self.position[1] - y_offset2)  # Bottom-right corner
        ]

        for corner in corners:
            if game_map.get_at((int(corner[0]), int(corner[1]))) == self.BORDER_COLOR:
                return True
        return False

    def rotate(self, direction):
        if direction == "LEFT":
            self.direction -= self.DIRECTION_CHANGE
            if self.direction < 0:
                self.direction += 2 * math.pi
        elif direction == "RIGHT":
            self.direction += self.DIRECTION_CHANGE
            if self.direction > 2 * math.pi:
                self.direction -= 2 * math.pi

    def speed_change(self, direction):
        if direction == "UP":
            self.speed += self.SPEED_CHANGE
            if self.speed > self.MAX_SPEED:
                self.speed = self.MAX_SPEED
        elif direction == "DOWN":
            self.speed -= self.SPEED_CHANGE
            if self.speed < self.MIN_SPEED:
                self.speed = self.MIN_SPEED

    def create_state(self, game_map):
        state = [(self.speed - self.MIN_SPEED) / (self.MAX_SPEED - self.MIN_SPEED)]
        max_distance = 200
        current_direction_delta = -(self.DIRECTION_CHECK_NUMBER - 1) / 2 * self.DIRECTION_CHECK_CHANGE
        for i in range(self.DIRECTION_CHECK_NUMBER):
            state.append(self.check_distance(current_direction_delta, max_distance, game_map) / max_distance)
            current_direction_delta += self.DIRECTION_CHECK_CHANGE
        return [state]

    def check_distance(self, direction_delta, distance_max, game_map):
        check_every = 10
        x_change = math.cos(self.direction + direction_delta) * check_every
        y_change = math.sin(self.direction + direction_delta) * check_every
        x = self.position[0]
        y = self.position[1]
        distance = 0
        while 0 < x < self.WIDTH and 0 < y < self.HEIGHT and game_map.get_at((int(x), int(y))) != self.BORDER_COLOR and distance < distance_max:
            x += x_change
            y -= y_change  # opposite direction
            distance += check_every
        return distance

    # now implemented methods from Enviroment Abstract
    def get_state(self):
        if self.next_state is None:
            self.next_state = self.create_state(self.game_map)
        return self.next_state

    def get_reward(self):
        return self.last_reward

    def react_to_action(self, action_index):
        self.use_action(action_index)
        self.update()
        self.last_reward = self.get_reward_car(self.game_map)
        return self.last_reward

    def is_alive(self):
        return self.alive

    def reset(self):
        self.position = [self.STARTX, self.STARTY]
        self.direction = self.START_DIRECTION
        self.speed = self.START_SPEED
        self.last_reward = 0
        self.alive = True
        self.next_state = None
