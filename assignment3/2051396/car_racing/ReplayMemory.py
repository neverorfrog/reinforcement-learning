from collections import namedtuple, deque
import random
from torchvision import transforms
import torch


class ReplayMemory:

    def __init__(self, capacity=50000, burn_in=10000, device=torch.device('cpu')):
        self.capacity = capacity
        self.burn_in = burn_in

        #Deques for storing
        self.transition = namedtuple('transition',('state', 'action', 'reward', 'done', 'next_state'))
        self.buffer = deque(maxlen=capacity)
        self.state = deque(maxlen=4)
        self.next_state = deque(maxlen=4)

        #Transforms for preprocessing before storing
        self.gs = transforms.Grayscale()
        self.rs = transforms.Resize((84,84))
        self.device = device

    def sample_batch(self, batch_size=32):
        transitions = random.sample(self.buffer,batch_size) #that's a list
        batch = self.transition(*zip(*transitions)) #that's a gigantic transition in which every element is actually a list
        return batch

    def preprocessing(self, observation):
        '''
        Input:
            a frame, i.e. a (96,96,3)~(height,width,channels) tensor 
        Output:
            the same frame, but with shape (1,84,84,1) greyscale and normalized
        '''
        observation = observation.transpose(2,0,1) #Torch wants images in format (channels, height, width)
        observation = torch.from_numpy(observation).float()
        observation = self.rs(observation) # resize
        observation = self.gs(observation) # grayscale
        return (observation/255) # normalize
    
    def update(self,observation,next_observation):
        '''
        param args: 
            observation(1 frame)
            next_observation(1 frame) 
        '''
        self.state.append(self.preprocessing(observation).to(self.device))
        self.next_state.append(self.preprocessing(next_observation).to(self.device))
    
    def getState(self):
        return self.buffer[-1].state

    def getNextState(self):
        return self.buffer[-1].next_state

    def store(self,observation,action,reward,done,next_observation):
        '''
        Add an experience
        :param args: 
            state(4 frames) already preprocessed, 
            action, 
            reward, 
            done, 
            next_state(4 frames) already preprocessed
        '''

        self.state.append(self.preprocessing(observation))
        self.next_state.append(self.preprocessing(next_observation))

        # print("Single observation"); print(self.state[-1].shape)#torch.Size([1, 84, 84])
        state = torch.stack([observation for observation in self.state],0).squeeze().to(self.device)
        # print("Stacked observations"); print(state.shape)#torch.Size([4, 84, 84])
        next_state = torch.stack([observation for observation in self.next_state],0).squeeze().to(self.device)

        self.buffer.append(self.transition(state,action,reward,done,next_state))

    def burn_in_capacity(self):
        return len(self.buffer) / self.burn_in

    def capacity(self):
        return len(self.buffer) / self.memory_size

    def __iter__(self):
       ''' Returns the Iterator object '''
       return iter(self.buffer)
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
        self.state.clear()
        self.next_state.clear()