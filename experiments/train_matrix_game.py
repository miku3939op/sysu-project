from algorithms.minimax_q import MiniMaxQLearner
from algorithms.policies import EpsGreedyQPolicy
from envs.matrix_game import MatrixGame
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nb_episode = 100

    agent1 = MiniMaxQLearner(aid=0, alpha=0.1, policy=EpsGreedyQPolicy(), actions=np.arange(2))  
    agent2 = MiniMaxQLearner(aid=1, alpha=0.1, policy=EpsGreedyQPolicy(), actions=np.arange(2))  

    game = MatrixGame()
    for episode in range(nb_episode):
        action1 = agent1.act()
        action2 = agent2.act()

        _, r1, r2 = game.step(action1, action2)

        agent1.observe(reward=r1, opponent_action=agent2.previous_action)
        agent2.observe(reward=r2, opponent_action=agent1.previous_action)
    print(agent1.pi)
    print(agent2.pi)
    # ipdb.set_trace()
    plt.plot(np.arange(len(agent1.pi_history)),agent1.pi_history, label="agent1's pi(0)")
    plt.plot(np.arange(len(agent2.pi_history)),agent2.pi_history, label="agent2's pi(0)")
    plt.ylim(0, 1)
    plt.xlabel("episode")
    plt.ylabel("pi(0)")
    plt.legend()
    plt.savefig("result.jpg")
    plt.show()
