import numpy as np
import matplotlib.pyplot as plt
import random

class GridWorld:
    def __init__(self, size=6, terminal_states=[(0, 1), (5, 5)]):
        self.size = size
        self.terminal_states = terminal_states
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.states = [(i, j) for i in range(size) for j in range(size)]

    def valid_pos(self, loc):
        row, col = loc
        return 0 <= row < self.size and 0 <= col < self.size

    def visualize_gridworld(self, policy=None, value_indic = False, policy_indic = False, iter = 10):
        gridworld = np.zeros((self.size, self.size), dtype=str)
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) in self.terminal_states:
                    gridworld[i, j] = 'T'
                elif policy is not None:
                    action = policy.get((i, j), 'X')
                    print((i,j ), ": ", action)
                    if action == (0, 1):
                        gridworld[i, j] = 'R'
                    elif action == (0, -1):
                        gridworld[i, j] = 'L'
                    elif action == (1, 0):
                        gridworld[i, j] = 'D'
                    elif action == (-1, 0):
                        gridworld[i, j] = 'U'
                    else:
                        gridworld[i, j] = 'X'
                else:
                    gridworld[i, j] = ' '
        print(gridworld)
        list_ac = [(1,0),(-1,0),(0,1),(0,-1)]
        arrows = {"R":(1,0), "L":(-1,0),"U":(0,1),"D":(0,-1), "X": random.choice(list_ac), "T": (0,0)}
        scale = 0.25
        fig, ax = plt.subplots(figsize=(6, 6))
        for r, row in enumerate(gridworld):
            for c, cell in enumerate(row):
                plt.arrow(c, 5-r, scale*arrows[cell][0], scale*arrows[cell][1], head_width=0.1)
        if value_indic:
            plt.title(f"Value Iteration for {iter} iterations")  
        elif policy_indic:
            plt.title(f"Policy Iteration for {iter} iterations")         
        plt.show()
        
    def visualize_values(self, V):
        matrix = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                matrix[i][j] = V[(i,j)]
                    
                
        fig, ax = plt.subplots()

        min_val, max_val = 0, 6

        for i in range(6):
            for j in range(6):
                c = matrix[i][j]
                print((i,j),":", c)
                ax.text(j+0.5,(5-i)+0.5,str(c), va='center', ha='center')

        #plt.matshow(matrix, cmap=plt.cm.Blues)

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xticks(np.arange(max_val))
        ax.set_yticks(np.arange(max_val))
        ax.grid()
        plt.show()
        

    def value_iteration(self, k=10, discount=1):
        V = {s: 0 for s in self.states}
        optimal_policy = {s: 0 for s in self.states}
        for _ in range(k):
            old_V = V.copy()
            for s in self.states:
                if s not in self.terminal_states:
                    Q = {}
                    for a in self.actions:
                        s_next = (s[0] + a[0], s[1] + a[1])
                        if self.valid_pos(s_next):
                            Q[a] = -1 + old_V.get(s_next,0)
                        else:
                            Q[a] = -1 + old_V[s]
                    V[s] = max(Q.values())
                    optimal_policy[s] = max(Q, key=Q.get)
        print(V)
        return V, optimal_policy

    def policy_iteration(self, k=10):
        policy = {s: random.choice(self.actions) for s in self.states}
        for _ in range(k):
            V = self.policy_evaluation(policy)
            policy = self.policy_improvement(V)
        return policy

    def policy_evaluation(self, policy, k=10):
        V = {s: 0 for s in self.states}
        for _ in range(k):
            old_V = V.copy()
            for s in self.states:
                if s not in self.terminal_states:
                    a = policy[s]
                    s_next = (s[0] + a[0], s[1] + a[1])
                    if self.valid_pos(s_next):
                        V[s] = -1 + old_V.get(s_next, 0)
                    else:
                        V[s] = -1 + old_V[s]
        return V

    def policy_improvement(self, V):
        policy = {s: random.choice(self.actions) for s in self.states}
        for s in self.states:
            if s not in self.terminal_states:
                Q = {}
                for a in self.actions:
                    s_next = (s[0] + a[0], s[1] + a[1])
                    if self.valid_pos(s_next):
                        Q[a] = -1 + V.get(s_next, 0)
                    else:
                        Q[a] = -1 + V[s]
                policy[s] = max(Q, key=Q.get)
        return policy
    def generate_episode(self,policy):
        episode = []
        init_state = random.choice(self.states)
        current_state = init_state
        step_cnt = 0
        while current_state not in self.terminal_states and step_cnt<50:
            action = policy[current_state]
            episode.append(((current_state,action),-1))
            s_next = (current_state[0] + action[0], current_state[1]+action[1])
            if self.valid_pos(s_next):                
                current_state = s_next
                #states.append(current_state)
            step_cnt+=1
            #print(episode)
        return episode  
    def return_val(self,indx,episode, discount = 1):
        sum = 0
        i = 0
        for elem in episode[indx:]:
            sum+= discount**i * elem[1]
            i+=1
        return sum
    def monte_carlo_first(self, iterations = 1000, discount_factor = 1):
        returns = {s: [] for s in self.states}
        V = {s:0 for s in self.states}
        policy = {s: random.choice(self.actions) for s in self.states}
        counts = {s:0 for s in self.states}
        for _ in range(iterations):
            past_states = []
            episode = self.generate_episode(policy)
            for i in range(len(episode)):
               # print(episode[i][0][0])
                #print(returns[episode[i][0][0]])
                if episode[i][0][0] not in past_states:
                    counts[episode[i][0][0]]+=1
                    returns[episode[i][0][0]].append(self.return_val(i,episode,discount_factor))
                    V[episode[i][0][0]] = round(sum(returns[episode[i][0][0]])/counts[episode[i][0][0]],2)
                    past_states.append(episode[i][0][0])
        return V
    def monte_carlo_every(self, iterations = 1000, discount_factor = 1):
        returns = {s: [] for s in self.states}
        V = {s:0 for s in self.states}
        policy = {s: random.choice(self.actions) for s in self.states}
        counts = {s:0 for s in self.states}
        for _ in range(iterations):
            #past_states = []
            episode = self.generate_episode(policy)
            for i in range(len(episode)):
                #print(episode[i][0][0])
                #print(returns[episode[i][0][0]])
                #if episode[i][0][0] not in past_states:
                counts[episode[i][0][0]]+=1
                returns[episode[i][0][0]].append(self.return_val(i,episode,discount_factor))
                V[episode[i][0][0]] = round(sum(returns[episode[i][0][0]])/counts[episode[i][0][0]],2)
                 #   past_states.append(episode[i][0][0])
        return V
        
    def TD_0(self, step_size = 0.5, iterations = 500, discount_factor = 1):
        policy = {s: random.choice(self.actions) for s in self.states}
        V = {s:0 for s in self.states}
        for _ in range(iterations):
            number_of_steps = 0
            s = random.choice(self.states)
            while s not in self.terminal_states and number_of_steps<50:
                number_of_steps+=1
                action = policy[s]
                s_next = (s[0] + action[0], s[1]+action[1])
                if self.valid_pos(s_next):
                    V[s] = round(V[s]+ step_size*(-1+discount_factor*V[s_next] - V[s]),1)
                    s = s_next
                else:
                    V[s] = round(V[s]+ step_size*(-1+discount_factor*V[s] - V[s]),1)
                    
        return V
                    

# Example usage:
gridworld = GridWorld()
#values_mc_first = gridworld.monte_carlo_first()
#values_mc_every = gridworld.monte_carlo_every()
values_td0 = gridworld.TD_0()
#gridworld.visualize_values(values_mc_first)
#gridworld.visualize_values(values_mc_every)
gridworld.visualize_values(values_td0)
"""value_and_policy = gridworld.value_iteration(k=10)
values = value_and_policy[0]
val_iter_opt = value_and_policy[1]
print(values)
pol_iter_opt = gridworld.policy_iteration(k=10)
gridworld.visualize_gridworld(policy=pol_iter_opt, policy_indic=True, iter = 10)
gridworld.visualize_gridworld(policy=val_iter_opt, value_indic=True, iter = 10)
#gridworld.visualize_values(values)"""

