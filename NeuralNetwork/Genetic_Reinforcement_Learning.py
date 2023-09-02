import numpy as np
import time
from NeuralNetwork.PPO_pytorch import PPO


class Genetic_Reinforcement_Learning():
    def __init__(self, reinforcement_learning_NN, enviroments, number_of_simulations=7, number_of_epochs_per_generation=50,
                 number_of_epochs_for_new_simulations=20,
                 number_of_removed_per_generation=2, combine_new_from=3, genetic_algorithm_session_ID=""):
        # input = [sequence_length, features] output = [action_dimmension]
        self.number_of_simulations = number_of_simulations
        self.number_of_epochs_per_generation = number_of_epochs_per_generation
        self.number_of_removed_per_generation = number_of_removed_per_generation
        self.number_of_epochs_for_new_simulations = number_of_epochs_for_new_simulations
        self.combine_new_from = combine_new_from
        self.genetic_algorithm_session_ID = genetic_algorithm_session_ID
        if genetic_algorithm_session_ID == "":
            self.genetic_algorithm_session_ID = str(time.time())

        self.reinforcement_learning_NN = reinforcement_learning_NN
        self.base_enviroments = enviroments
        self.population = []
        self.generationCounter = 1
        self.inGenerationCounter = 1
        for i in range(number_of_simulations):
            self.population.append(self.create_simulation())
            self.inGenerationCounter += 1
        self.inGenerationCounter = 0

    def create_simulation(self):
        self.reinforcement_learning_NN.reset_with_new_weights()
        return self.Simulation(self.reinforcement_learning_NN, self.base_enviroments,
                               name="gen_" + str(self.generationCounter) + "_num" + str(self.inGenerationCounter),
                               genetic_algorithm_session_ID=self.genetic_algorithm_session_ID)

    def run_generation(self):
        self.generationCounter += 1
        self.run_learning_routine()
        self.create_new_simulations()
        self.sort_population(ascending=False)
        best_simulation = self.population[0]
        self.print_population_stats()
        #self.reinforcement_learning_NN.set_weights(best_simulation.current_Reinforcement_Learning.weights)
        self.reinforcement_learning_NN.load_weights(best_simulation.best_Reinforcement_Learning.weights)
        for environment in self.base_enviroments:
            environment.reset()

    def print_population_stats(self):
        print("\ngeneration {}.".format(self.generationCounter))
        for simulation in self.population:
            print("\t{} - current: {} - best: {}".format(simulation.name,
                                                         simulation.current_Reinforcement_Learning.total_reward,
                                                         simulation.best_Reinforcement_Learning.total_reward))

    def create_new_simulations(self):
        self.sort_population(ascending=True)
        total_rewards = [individual.best_Reinforcement_Learning.total_reward for individual in self.population]
        total_rewards = total_rewards[self.number_of_removed_per_generation:]
        probabilities = self.softmax(total_rewards)

        self.inGenerationCounter = 0
        for i in range(self.number_of_removed_per_generation):
            self.inGenerationCounter += 1
            indices = np.random.choice(len(probabilities), size=self.combine_new_from, replace=False, p=probabilities)

            new_simulation = self.create_simulation()
            new_simulation.session_start()

            states, actions, expected_rewards = None, None, None
            for old_simulation in [self.population[index + self.number_of_removed_per_generation] for index in indices]:
                if states is None:
                    states = np.array(old_simulation.best_Reinforcement_Learning.trajectories['states'])
                    actions = np.array(old_simulation.best_Reinforcement_Learning.trajectories['action_indexes'])
                    expected_rewards = np.array(old_simulation.best_Reinforcement_Learning.trajectories['expected_rewards'])
                else:
                    states = np.concatenate((states, old_simulation.best_Reinforcement_Learning.trajectories['states']), axis=0)
                    actions = np.concatenate((actions, old_simulation.best_Reinforcement_Learning.trajectories['action_indexes']), axis=0)
                    expected_rewards = np.concatenate(
                        (expected_rewards, old_simulation.best_Reinforcement_Learning.trajectories['expected_rewards']), axis=0)

            new_simulation.Reinforcement_Learning_NN.train_categorical_crossentropy(states, actions, expected_rewards,
                                                                                    expected_rewards=expected_rewards,
                                                                                    epochs_actor=self.number_of_epochs_for_new_simulations,
                                                                                    epochs_critic=self.number_of_epochs_for_new_simulations)
            new_simulation.current_Reinforcement_Learning.save_weights(new_simulation.Reinforcement_Learning_NN)
            new_simulation.best_Reinforcement_Learning.copy_attributes(new_simulation.current_Reinforcement_Learning)
            new_simulation.best_Reinforcement_Learning.save_weights(new_simulation.Reinforcement_Learning_NN)
            new_simulation.session_end()

            self.population[i] = new_simulation

    def sort_population(self, ascending=True):
        self.population.sort(key=lambda x: x.best_Reinforcement_Learning.total_reward, reverse=not ascending)

    def run_learning_routine(self):
        print("running learning routine")
        for simulation in self.population:
            simulation.session_start()
            for i in range(self.number_of_epochs_per_generation):
                simulation.run_one_epoch_learning()
            simulation.session_end()

    class Simulation():
        def __init__(self, Reinforcement_Learning_NN, enviroments, name="", genetic_algorithm_session_ID=""):
            self.Reinforcement_Learning_NN = Reinforcement_Learning_NN
            self.name = name
            self.file_name = "simulation_" + genetic_algorithm_session_ID + "_" + self.name
            self.best_Reinforcement_Learning = self.Reinforcement_Learning_Data(self.Reinforcement_Learning_NN, self.file_name + "_best")
            self.current_Reinforcement_Learning = self.Reinforcement_Learning_Data(self.Reinforcement_Learning_NN, self.file_name + "_current")
            self.choose_always_best = False
            self.simulation_enviroments = []
            for enviroment in enviroments:
                self.simulation_enviroments.append(self.simulation_enviroment(enviroment, self.Reinforcement_Learning_NN))

        def session_start(self):
            self.Reinforcement_Learning_NN.load_weights(self.current_Reinforcement_Learning.weights)

        def run_one_epoch_learning(self):
            self.run_one_exploration_step_learning_state()

            self.train_RNN(self.current_Reinforcement_Learning.trajectories)
            self.run_one_exploration_step_production_state()

        def run_one_exploration_step_learning_state(self):
            for enviroment in self.simulation_enviroments:
                enviroment.run_one_epoch()
            self.actualise_current_Reinforcement_Learning()
            # if self.current_Reinforcement_Learning.total_reward >= self.best_Reinforcement_Learning.total_reward:
            #    self.best_Reinforcement_Learning.copy_attributes(self.current_Reinforcement_Learning)

        def run_one_exploration_step_production_state(self):
            for enviroment in self.simulation_enviroments:
                enviroment.run_one_epoch(False)
            self.actualise_current_Reinforcement_Learning()
            if self.current_Reinforcement_Learning.total_reward >= self.best_Reinforcement_Learning.total_reward:
                self.best_Reinforcement_Learning.copy_attributes(self.current_Reinforcement_Learning)
                self.best_Reinforcement_Learning.save_weights(self.Reinforcement_Learning_NN)

        def session_end(self):
            self.run_one_exploration_step_production_state()

        class Reinforcement_Learning_Data():
            def __init__(self, Reinforcement_Learning_NN, file_name):
                self.trajectories = {}
                self.total_reward = 0
                self.file_name = file_name
                self.save_weights(Reinforcement_Learning_NN)

            def save_weights(self, RNN):
                self.weights = RNN.save_weights(self.file_name)

            def set_weights(self, RNN):
                RNN.set_weights(self.weights)

            def copy_attributes(self, Reinforcement_Learning_Data):
                self.trajectories = Reinforcement_Learning_Data.trajectories
                self.total_reward = Reinforcement_Learning_Data.total_reward

        def actualise_current_Reinforcement_Learning(self):
            total_reward = 0
            states, action_indexes, rewards, expected_rewards, advantages = None, None, None, None, None

            for enviroment in self.simulation_enviroments:
                if states is None:
                    states = np.array(enviroment.states)
                    action_indexes = np.array(enviroment.action_indexes)
                    rewards = np.array(enviroment.rewards)
                    expected_rewards = np.array(self.Reinforcement_Learning_NN.get_expected_rewards(enviroment.states,
                                                                                                    enviroment.rewards))
                    advantages = np.array(
                        self.Reinforcement_Learning_NN.get_Advantages(enviroment.states, enviroment.rewards))
                else:
                    states = np.concatenate((states, np.array(enviroment.states)), axis=0)
                    action_indexes = np.concatenate((action_indexes, np.array(enviroment.action_indexes)), axis=0)
                    rewards = np.concatenate((rewards, np.array(enviroment.rewards)), axis=0)
                    expected_rewards = np.concatenate((expected_rewards, np.array(
                        self.Reinforcement_Learning_NN.get_expected_rewards(enviroment.states))), axis=0)
                    advantages = np.concatenate((advantages, np.array(
                        self.Reinforcement_Learning_NN.get_Advantages(enviroment.states, enviroment.rewards))), axis=0)
                total_reward += enviroment.get_reward_sum()
                enviroment.reset()

            trajectories = {'states': states, 'action_indexes': action_indexes, 'rewards': rewards,
                            'expected_rewards': expected_rewards, 'advantages': advantages}
            self.current_Reinforcement_Learning.trajectories = trajectories
            self.current_Reinforcement_Learning.total_reward = total_reward
            self.current_Reinforcement_Learning.save_weights(self.Reinforcement_Learning_NN)

        def train_RNN(self, trajectories):
            self.Reinforcement_Learning_NN.train(states=trajectories['states'], actions_indexes=trajectories['action_indexes'],
                                                 rewards=trajectories['rewards'], action_probabilities=None,
                                                 expected_rewards=trajectories['expected_rewards'],
                                                 advantages=trajectories['advantages'])

        class simulation_enviroment():
            def __init__(self, enviroment, Reinforcement_Learning_NN):
                self.enviroment = enviroment
                self.Reinforcement_Learning_NN = Reinforcement_Learning_NN
                self.states = []
                self.action_indexes = []
                self.rewards = []
                self.is_alive = True

            def step(self, learning_state=True):
                if self.is_alive:
                    enviroment_state = self.enviroment.get_state()
                    self.states.append(enviroment_state)
                    action_probs = self.Reinforcement_Learning_NN.get_action_probs(enviroment_state)
                    if learning_state:
                        action_index = self.enviroment.get_action_learning_state(action_probs)
                    else:
                        action_index = self.enviroment.get_action_production_state(action_probs)
                    self.action_indexes.append(action_index)
                    self.enviroment.react_to_action(action_index)
                    self.rewards.append(self.enviroment.get_reward())
                    self.is_alive = self.enviroment.is_alive()

            def reset(self):
                self.enviroment.reset()
                self.states = []
                self.action_indexes = []
                self.rewards = []
                self.is_alive = True

            def run_one_epoch(self, learning_state=True):
                self.reset()
                while self.is_alive:
                    self.step(learning_state)

            def get_reward_sum(self):
                return sum(self.rewards)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=0)