import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, Flatten
from keras import backend as K


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


class PPO:
    def __init__(self, state_length, state_features, action_dim, entropy_factor=0.01, epsilon=0.2,
                 gamma=0.99, lamda_tradeof=0.95, actor_learning_rate=0.001, critic_learning_rate=0.002,
                 actor_epochs=10, critic_epochs=10, batch_size=32):
        self.state_length = state_length
        self.state_features = state_features
        self.action_dim = action_dim
        self.entropy_factor = entropy_factor
        self.epsilon = epsilon
        self.gamma = gamma
        self.lamda_tradeof = lamda_tradeof
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_epochs = actor_epochs
        self.critic_epochs = critic_epochs
        self.batch_size = batch_size
        self.current_loss = "actor_loss"

        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def get_weights(self):
        weights = {
            'actor': self.actor.get_weights(),
            'critic': self.critic.get_weights(),
            'actor_optimizer': self.actor.optimizer.get_weights(),
            'critic_optimizer': self.critic.optimizer.get_weights()
        }
        return weights

    def set_weights(self, weights):
        self.current_loss = "actor_loss"
        self.actor.set_weights(weights['actor'])
        self.critic.set_weights(weights['critic'])
        self.actor.optimizer.set_weights(weights['actor_optimizer'])
        self.critic.optimizer.set_weights(weights['critic_optimizer'])

    def reset_with_new_weights(self):
        self.current_loss = "actor_loss"
        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        self._states_input = Input(shape=(self.state_length, self.state_features))
        self._advantage_input = Input(shape=(1,))
        self._old_prediction_input = Input(shape=(self.action_dim,))

        x = Dense(64, activation='relu')(self._states_input)
        x = Dense(32, activation='relu')(x)
        x = Flatten()(x)
        probs = Dense(self.action_dim, activation='softmax')(x)

        actor = Model(inputs=[self._states_input, self._advantage_input, self._old_prediction_input], outputs=[probs])
        actor.compile(loss=self.actor_loss(self._advantage_input, self._old_prediction_input), optimizer=tf.optimizers.Adam(lr=self.actor_learning_rate))
        #actor.summary()
        actor.fit([np.zeros((1, self.state_length, self.state_features)), np.zeros(1),
                  np.zeros((1, self.action_dim))], np.zeros((1, self.action_dim)),
                  verbose=0)  # optimizer has to initialise weights
        return actor

    def actor_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction

            r = prob / (old_prob + 1e-10)
            clipped_r = K.clip(r, min_value=1 - self.epsilon, max_value=1 + self.epsilon)
            entropy_bonus = self.entropy_factor * (prob * K.log(prob + 1e-10))
            return -K.mean(K.minimum(r * advantage, clipped_r * advantage) + entropy_bonus)
        return loss

    def build_critic(self):
        state = Input(shape=(self.state_length, self.state_features))
        x = Dense(64, activation='relu')(state)
        x = Dense(32, activation='relu')(x)
        x = Flatten()(x)
        value = Dense(1)(x)
        critic = Model(inputs=[state], outputs=[value])
        critic.compile(loss='mse', optimizer=tf.optimizers.Adam(lr=self.critic_learning_rate))
        #critic.summary()
        critic.fit([np.zeros((1, self.state_length, self.state_features))], [np.zeros(1)],
                   verbose=0)  # optimizer has to initialise weights
        return critic

    def get_action(self, state):
        state = np.array([state])
        action_prob = self.actor.predict([state, np.zeros((1, 1)), np.zeros((1, self.action_dim))])
        action = np.random.choice(self.action_dim, p=action_prob[0])
        return action

    def get_action_highest_prob(self, state):
        state = np.array([state])
        action_prob = self.actor.predict([state, np.zeros((1, 1)), np.zeros((1, self.action_dim))])
        action = np.argmax(action_prob[0])
        return action

    def train(self, states, actions_indexes, rewards, action_probabilities=None, expected_rewards=None, advantages=None):
        self.ensure_loss('actor_loss')
        states = np.array(states)
        actions_indexes = np.array(actions_indexes)
        rewards = np.array(rewards, dtype=np.float32)
        if action_probabilities is None:
            action_probabilities = self.actor.predict(
                [states, np.zeros((len(states), 1)), np.zeros((len(states), self.action_dim))])

        if expected_rewards is None:
            expected_rewards = self.get_expected_rewards(states, rewards)

        if advantages is None:
            advantages = self.get_Advantages(states, rewards)

        actions_onehot = tf.keras.utils.to_categorical(actions_indexes, num_classes=self.action_dim, dtype=np.float32)
        actions_onehot = np.array(actions_onehot, dtype=np.float32)

        self.actor.fit([states, advantages, action_probabilities], [actions_onehot], verbose=0, epochs=self.actor_epochs, batch_size=self.batch_size, shuffle=True)
        self.critic.fit(states, expected_rewards, verbose=0, epochs=self.critic_epochs, batch_size=self.batch_size, shuffle=True)

    def train_categorical_crossentropy(self, states, actions_indexes, rewards, expected_rewards=None, epochs_actor=10, epochs_critic=10):
        self.ensure_loss('categorical_crossentropy')
        states = np.array(states)
        actions_indexes = np.array(actions_indexes)
        rewards = np.array(rewards, dtype=np.float32)

        if expected_rewards is None:
            expected_rewards = self.get_expected_rewards(states, rewards, use_last_value=False)
        actions_onehot = tf.keras.utils.to_categorical(actions_indexes, num_classes=self.action_dim, dtype=np.float32)
        actions_onehot = np.array(actions_onehot, dtype=np.float32)

        self.actor.fit([states, np.zeros((len(states), 1)), np.zeros((len(states), self.action_dim))],
                       [actions_onehot], verbose=0, epochs=epochs_actor, batch_size=self.batch_size, shuffle=True)
        self.critic.fit([states], [expected_rewards], verbose=0, epochs=epochs_critic, batch_size=self.batch_size, shuffle=True)

    def ensure_loss(self, target_loss):
        if self.current_loss != target_loss:
            if target_loss == 'categorical_crossentropy':
                self.actor.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=self.actor_learning_rate))
            elif target_loss == 'actor_loss':
                self.actor.compile(loss=self.actor_loss(self._advantage_input, self._old_prediction_input),
                                   optimizer=tf.optimizers.Adam(lr=self.actor_learning_rate))
            self.current_loss = target_loss

    def get_expected_rewards(self, states, rewards, use_last_value=True):
        state = np.array([states[-1]])
        value = self.critic.predict(state)[0] if use_last_value else 0.0
        expected_rewards = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            value = rewards[t] + self.gamma * value
            expected_rewards[t] = value
        return expected_rewards

    def get_Advantages(self, states, rewards):
        states = np.array(states)
        values = self.critic.predict(states)
        advantages = np.zeros_like(rewards)
        gae = 0
        next_values = values[-1][0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values - values[t][0]
            gae = delta + self.gamma * self.lamda_tradeof * gae
            advantages[t] = gae
            next_values = values[t]
        return np.array(advantages)
