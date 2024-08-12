import gym
import numpy as np
import random
from skimage.color import rgb2gray
from skimage.transform import resize

class QLearningAgent:
    def __init__(self, action_space, epsilon=0.1, alpha=0.5, gamma=0.99):
        """
        Initializes the Q-learning agent with the given parameters.
        - epsilon: exploration rate
        - alpha: learning rate
        - gamma: discount factor
        - action_space: the space of possible actions the agent can take
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.qValues = {}
        self.action_space = action_space

    def getQValue(self, state, action):
        """
        Returns the Q-value for a given state-action pair.
        If the state-action pair has not been encountered before, returns 0.0.
        """
        return self.qValues.get((tuple(state), action), 0.0)

    def computeValueFromQValues(self, state):
        """
        Computes the maximum Q-value for all possible actions in a given state.
        This represents the best possible future reward the agent can expect
        from this state.
        """
        actions = range(self.action_space.n)
        return max([self.getQValue(state, action) for action in actions])

    def computeActionFromQValues(self, state):
        """
        Determines the best action to take in a given state based on the Q-values.
        In case of ties (multiple actions with the same Q-value), one action is chosen at random.
        """
        actions = range(self.action_space.n)
        maxValue = float('-inf')
        bestActions = []
        for action in actions:
            qVal = self.getQValue(state, action)
            if qVal > maxValue:
                maxValue = qVal
                bestActions = [action]
            elif qVal == maxValue:
                bestActions.append(action)
        return random.choice(bestActions)

    def getAction(self, state):
        """
        Chooses the action to take in the current state.
        Using the probability epsilon, a random action is selected.
        Otherwise, the best action based on the Q-values is chosen.
        Additionally, the agent avoids shooting if a spaceship is detected above.
        """
        if self.is_spaceship_above(state) and self.action_space.contains(0):
            possible_actions = [a for a in range(self.action_space.n) if a != 0]
            if random.random() < self.epsilon:
                return random.choice(possible_actions)
            else:
                return self.computeActionFromQValues(state)
        else:
            if random.random() < self.epsilon:
                return self.action_space.sample()
            else:
                return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
        Updates the Q-value for a given state-action pair based on the observed transition.
        The Q-value is updated using the Q-learning update rule, incorporating the reward received
        and the maximum expected future reward from the next state.
        """
        if self.was_hit(state, nextState):
            reward -= 1.0  # Penalty for getting hit
        else:
            reward += 0.1  # Small reward for surviving

        if action in [2, 3]:  # Encourage movement actions (left/right)
            movement_reward = 0.1
        else:
            movement_reward = -0.1  # Discourage non-movement actions
        
        reward += movement_reward

        state_tuple = tuple(state)
        nextState_tuple = tuple(nextState)
        
        # Q-learning update formula
        sample = reward + self.gamma * self.computeValueFromQValues(nextState)
        self.qValues[(state_tuple, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample

    def is_spaceship_above(self, state):
        """
        Detects if a spaceship is directly above the agent based on the pixel intensity.
        This is used to avoid shooting at spaceships.
        """
        state_image = np.array(state).reshape((84, 84))
        threshold_intensity = 0.85  # Intensity threshold to detect the spaceship
        if np.mean(state_image[0:20, 40:44]) > threshold_intensity:
            return True
        return False

    def was_hit(self, state, nextState):
        """
        Detects if the agent was hit by comparing the current state and the next state.
        A significant decrease in the pixel sum suggests the agent was hit.
        """
        state_image = np.array(state).reshape((84, 84))
        next_state_image = np.array(nextState).reshape((84, 84))
        threshold = 30  
        hit_detected = np.sum(next_state_image) < np.sum(state_image) - threshold
        return hit_detected

    def saveQValues(self, filename):
        """
        Saves the current Q-values to a file, allowing the agent's learned policy
        to be reloaded later.
        """
        np.save(filename, self.qValues)

    def loadQValues(self, filename):
        """
        Loads Q-values from a file, restoring the agent's learned policy.
        """
        self.qValues = np.load(filename, allow_pickle=True).item()

def preprocess_state(state):
    """
    Converts the game screen to a grayscale image, resizes it,
    and flattens it into a 1D array to be used as input for the Q-learning agent.
    """
    if isinstance(state, (tuple, list)):
        state = state[0]
    gray = rgb2gray(state)
    resized = resize(gray, (84, 84))
    return resized.flatten()

def train(agent, env, episodes=1000, max_steps=5000):
    """
    Trains the Q-learning agent over a specified number of episodes.
    In each episode, the agent interacts with the environment for a maximum number of steps.
    The agent's Q-values are updated based on its experiences.
    Q-values are saved every 10 episodes.
    """
    for episode in range(episodes):
        state = preprocess_state(env.reset())
        total_reward = 0

        for t in range(max_steps):
            env.render()  # Render the environment
            action = agent.getAction(state)
            result = env.step(action)

            if len(result) == 4:
                next_state, reward, done, _ = result
            elif len(result) == 5:
                next_state, reward, done, truncated, _ = result
                done = done or truncated
            else:
                raise ValueError(f"Unexpected number of values returned by env.step(): {len(result)}")

            next_state = preprocess_state(next_state)
            agent.update(state, action, next_state, reward)
            total_reward += reward
            state = next_state

            if done:
                break

        agent.epsilon = max(0.01, agent.epsilon * 0.995)  # Epsilon reduces exploration over time

        print(f"Episode {episode + 1}/{episodes} finished with reward: {total_reward}")
        if (episode + 1) % 10 == 0:  # Save Q-values every 10 episodes
            agent.saveQValues(f'q_values_{episode + 1}.npy')

if __name__ == "__main__":
    # Create the Space Invaders environment and train the Q-learning agent
    env = gym.make('SpaceInvaders-v4', render_mode='human')
    agent = QLearningAgent(env.action_space)
    train(agent, env)
