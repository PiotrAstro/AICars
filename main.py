import math
import pygame
import sys

from Car_AI.Car_Enviroment import Car_Enviroment
from NeuralNetwork.Genetic_Reinforcement_Learning import Genetic_Reinforcement_Learning
from NeuralNetwork.PPO_implementation import PPO

# Constants
WIDTH, HEIGHT = 800, 800
CAR_SIZE_X, CAR_SIZE_Y = 46, 27
BORDER_COLOR = (255, 255, 255)
STARTX, STARTY = 400, 730
START_SPEED = 2
MAX_SPEED, MIN_SPEED = 10, 1
SPEED_CHANGE = 1
START_DIRECTION = math.radians(0)
DIRECTION_CHANGE = math.radians(1)
DIRECTION_CHECK_NUMBER = 9
DIRECTION_CHECK_CHANGE = math.radians(45)

class Car:
    def __init__(self, enviroment, RNN):
        self.sprite = pygame.image.load('car.png').convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.enviroment = enviroment
        self.RNN = RNN
        self.reset()

    def reset(self):
        self.enviroment.reset()
        self.alive = True

    def getDrawPosition(self):
        rotated_sprite = pygame.transform.rotate(self.sprite, math.degrees(self.enviroment.direction))
        draw_position = (
            self.enviroment.position[0] - rotated_sprite.get_width() / 2, self.enviroment.position[1] - rotated_sprite.get_height() / 2)
        return rotated_sprite, draw_position

    def draw(self, screen):
        rotated_sprite, draw_position = self.getDrawPosition()
        screen.blit(rotated_sprite, draw_position)

    def step(self):
        state = self.enviroment.get_state()
        action = self.RNN.get_action(state)
        self.enviroment.react_to_action(action)
        self.alive = self.enviroment.is_alive()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
# Load the map
game_map = pygame.image.load('map2.png').convert()
game_map = pygame.transform.scale(game_map, (WIDTH, HEIGHT))

enviroment = Car_Enviroment(
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
                 DIRECTION_CHECK_CHANGE=math.radians(45))
rnn = PPO(1, DIRECTION_CHECK_NUMBER + 1, 3)

car = Car(enviroment, rnn)
genetic_rl = Genetic_Reinforcement_Learning(rnn, [enviroment])

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # keys = pygame.key.get_pressed()
    car.step( )
    if not car.alive:
        car.reset()
        genetic_rl.run_generation()

    screen.fill((0, 0, 0))  # Clear the screen
    screen.blit(game_map, (0,0))  # Draw the map
    car.draw(screen)  # Draw the car on the screen

    pygame.display.flip()  # Update the display