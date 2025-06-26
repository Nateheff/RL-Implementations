import numpy as np
import torch
from collections import deque
"""
We prioritze replays in the replay memory by TD Error (r + yV(S') - V(s))

IMPLEMENTATION:
We need to store priorities and be able to access transitions with the highest priorities
We have an array of priorities
We have an array of transitions
We maintain both arrays such that priority at index i corresponds to the transition at i

"""

class PrioritizedMemory():
    def __init__(self, transitions):
        self.transitions = transitions
        self.priorities = torch.zeros(transitions)
        self.D = []
        self.count = 0

    def add(self, transition, priority):

        if self.count < self.transitions:

            self.priorities[self.count] = priority
            self.D.append(transition)
            self.count += 1
        else:
            
            remove_idx = self.priorities.argmin().item()
            self.priorities[remove_idx] = priority
            self.D[remove_idx] = transition

    def get(self):
        if self.count <= 4:
            return False
        probabilities = torch.softmax(self.priorities, dim=0)
        # print(self.count, self.priorities, probabilities)
        samples = torch.multinomial(probabilities[:self.count], num_samples=4)
        randoms = [self.D[sample] for sample in samples]
        return randoms, samples
    
    def update(self, idx, value):
        self.priorities[idx] = value

    @property
    def next_idx(self):
        if self.count < self.transitions:
            return self.count
        else:
            return self.priorities.argmin().item()
