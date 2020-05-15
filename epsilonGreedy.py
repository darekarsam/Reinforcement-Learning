import numpy as np

NO_OF_TRIALS = 10000
EPS = 0.1
BANDIT_PROB = [0.2,0.4,0.6] #considering 3 bandits

class Bandit:

    def __init__(self, p):
        self.p = p
        self.pEstimate = 0
        self.N = 0.00001   #avoiding divide by zero if bandit never selected

    def pull(self):
        return np.random.random() < self.p
         
    def update(self, currentReward):
        self.N += 1
        self.pEstimate = (self.pEstimate + currentReward) / self.N


def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROB]
    rewards = np.zeros(NO_OF_TRIALS)
    explored = 0
    exploited = 0
    optimal = 0

    # get bandit with max reward
    optimalBanditIndex = np.argmax([bandit.p for bandit in bandits])

    for i in range(NO_OF_TRIALS):
        # use epsilon greedy
        if np.random.random() <= EPS:
            explored += 1
            # select any bandit randomly
            banditIndex = np.random.randint(0,len(bandits)) 
        else:
            exploited += 1
            # select optimal bandit
            banditIndex = optimalBanditIndex

        if banditIndex == optimalBanditIndex:
            optimal += 1
        
        reward = bandits[banditIndex].pull()
        # pull arm of the selected bandit and update reward
        rewards[i] = reward

        # update distribution of bandit pulled 
        bandits[banditIndex].update(reward)

    for i in bandits:
        print(i.pEstimate)

    print('No of times Explored: {}'.format(explored))
    print('No of times Exploited: {}'.format(exploited))
    print('No of times optimal selected: {}'.format(optimal))

if __name__ == "__main__":
    experiment()