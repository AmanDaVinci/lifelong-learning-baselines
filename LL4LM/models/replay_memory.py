import random


class ReplayMemory():

    def __init__(self, first_batch):
        self.memory = [first_batch]
    
    def __len__(self):
        return len(self.memory)
    
    def add(self, batch, probability):
        if probability > random.random(): 
            self.memory.append(batch)
    
    def sample(self, num_batches):
        return random.choices(self.memory, k=num_batches)

