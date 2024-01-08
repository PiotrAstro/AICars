import math

import numpy as np
import pygame
import sys

from NeuralNetwork.PPO_implementation import PPO

# Constants
WIDTH, HEIGHT = 800, 800
CAR_SIZE_X, CAR_SIZE_Y = 46, 27
BORDER_COLOR = (255, 255, 255)
STARTX, STARTY = 400, 730
START_SPEED = 1
MAX_SPEED, MIN_SPEED = 10, 1
SPEED_CHANGE = 1
START_DIRECTION = math.radians(0)
DIRECTION_CHANGE = math.radians(1)
DIRECTION_CHECK_NUMBER = 9
DIRECTION_CHECK_CHANGE = math.radians(45)

class Car:
    def __init__(self):
        self.sprite = pygame.image.load('car.png').convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.GENERATION = 1
        self.rewards = []
        self.best_reward = 0
        self.reset()
        self.PPO = PPO(1, DIRECTION_CHECK_NUMBER + 1, 3)

    def reset(self):
        print("g." + str(self.GENERATION))
        self.GENERATION += 1
        sum_of_rewards = sum(self.rewards)

        print("\tsum of rewards: " + str(sum_of_rewards))
        print("\tbest reward: " + str(self.best_reward))

        if sum_of_rewards > self.best_reward:
            self.best_reward = sum_of_rewards

        self.position = [STARTX, STARTY]
        self.direction = START_DIRECTION
        self.speed = START_SPEED
        self.alive = True
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_state = None

    def getDrawPosition(self):
        rotated_sprite = pygame.transform.rotate(self.sprite, math.degrees(self.direction))
        draw_position = (
            self.position[0] - rotated_sprite.get_width() / 2, self.position[1] - rotated_sprite.get_height() / 2)
        return rotated_sprite, draw_position

    def draw(self, screen):
        rotated_sprite, draw_position = self.getDrawPosition()
        screen.blit(rotated_sprite, draw_position)

    def step(self, game_map):
        if self.next_state is None:
            self.next_state = self.create_state(game_map)
        self.states.append(self.next_state)
        action = self.PPO.get_action(self.next_state)
        self.actions.append(action)

        self.use_action(action)
        self.update()

        reward = self.get_reward(game_map)
        self.rewards.append(reward)

        # Check for collision
        if not self.alive:
            self.PPO.train(self.states, self.actions, self.rewards)
            self.reset()

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


    def get_reward(self, game_map):
        if self.check_collision(game_map):
            self.alive = False
            return -5
        else:
            self.next_state = self.create_state(game_map)
            minimum_distance = 100
            for i in range(DIRECTION_CHECK_NUMBER):
                if self.next_state[0][i + 1] < minimum_distance:
                    minimum_distance = self.next_state[0][i + 1]
            returnValue = minimum_distance + (self.next_state[0][0]) / 10
            return returnValue

    def update(self):
        self.position[0] += math.cos(self.direction) * self.speed
        self.position[1] -= math.sin(self.direction) * self.speed  # y-axis is inverted


    def check_collision(self, game_map):
        cos_theta = math.cos(self.direction)
        sin_theta = math.sin(self.direction)

        x_half = CAR_SIZE_X / 2
        y_half = CAR_SIZE_Y / 2

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
            if game_map.get_at((int(corner[0]), int(corner[1]))) == BORDER_COLOR:
                return True
        return False

    def rotate(self, direction):
        if direction == "LEFT":
            self.direction -= DIRECTION_CHANGE
            if self.direction < 0:
                self.direction += 2 * math.pi
        elif direction == "RIGHT":
            self.direction += DIRECTION_CHANGE
            if self.direction > 2 * math.pi:
                self.direction -= 2 * math.pi

    def speed_change(self, direction):
        if direction == "UP":
            self.speed += SPEED_CHANGE
            if self.speed > MAX_SPEED:
                self.speed = MAX_SPEED
        elif direction == "DOWN":
            self.speed -= SPEED_CHANGE
            if self.speed < MIN_SPEED:
                self.speed = MIN_SPEED

    def create_state(self, game_map):
        state = [(self.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)]
        max_distance = 200
        current_direction_delta = -(DIRECTION_CHECK_NUMBER - 1) / 2 * DIRECTION_CHECK_CHANGE
        for i in range(DIRECTION_CHECK_NUMBER):
            state.append(self.check_distance(current_direction_delta, max_distance, game_map) / max_distance)
            current_direction_delta += DIRECTION_CHECK_CHANGE
        return [state]

    def check_distance(self, direction_delta, distance_max, game_map):
        check_every = 10
        x_change = math.cos(self.direction + direction_delta) * check_every
        y_change = math.sin(self.direction + direction_delta) * check_every
        x = self.position[0]
        y = self.position[1]
        distance = 0
        while 0 < x < WIDTH and 0 < y < HEIGHT and game_map.get_at((int(x), int(y))) != BORDER_COLOR and distance < distance_max:
            x += x_change
            y -= y_change  # opposite direction
            distance += check_every
        return distance


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

car = Car()

# Load the map
game_map = pygame.image.load('map2.png').convert()
game_map = pygame.transform.scale(game_map, (WIDTH, HEIGHT))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # keys = pygame.key.get_pressed()
    car.step(game_map)

    screen.fill((0, 0, 0))  # Clear the screen
    screen.blit(game_map, (0,0))  # Draw the map
    car.draw(screen)  # Draw the car on the screen

    pygame.display.flip()  # Update the display