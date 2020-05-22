import numpy as np
import matplotlib.pyplot as plt

NO_OF_TRIALS = 10000
EPS = 0.1
BANDIT_PROB = [0.2,0.4,0.6] #considering 3 bandits

class Bandit:

    def __init__(self, p):
        self.p = p
        self.pEstimate = 0.9
        self.N = 1.000001   #avoiding divide by zero if bandit never selected

    def pull(self):
        # Draw 1 with probability p
        return np.random.random() < self.p
         
    def update(self, currentReward):
        self.N += 1
        self.pEstimate = ((self.N - 1)*self.pEstimate + currentReward) / self.N


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROB]
    rewards = np.zeros(NO_OF_TRIALS)

    for i in range(NO_OF_TRIALS):
        # select optimal bandit
        banditIndex = np.argmax([bandit.pEstimate for bandit in bandits])

        
        reward = bandits[banditIndex].pull()
        # pull arm of the selected bandit and update reward
        rewards[i] = reward

        # update distribution of bandit pulled 
        bandits[banditIndex].update(reward)

    for i, bandit in enumerate(bandits):
        print('mean estimate of {}: {}'.format(i, bandit.pEstimate))
    
    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NO_OF_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NO_OF_TRIALS) * np.max(BANDIT_PROB))
    plt.show()

if __name__ == "__main__":
    experiment()