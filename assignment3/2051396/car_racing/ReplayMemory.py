from collections import namedtuple, deque
import random
from torchvision import transforms
import torch


class UniformER:

    def __init__(self, env, n_frames, capacity=50000, burn_in=10000, device=torch.device('cpu')):
        self.capacity = capacity
        self.burn_in = burn_in

        #Deques for storing
        self.transition = namedtuple('transition',('state', 'action', 'reward', 'done', 'next_state'))
        self.buffer = deque(maxlen=capacity)
        self.state = deque(maxlen=n_frames)
        self.next_state = deque(maxlen=n_frames)
        self.n_frames = n_frames

        #Transforms for preprocessing before storing
        self.gs = transforms.Grayscale()
        self.rs = transforms.Resize((84,84))
        self.device = device

        #I want to construct it with an already full state (with noop transitions)
        env.reset()
        observation,reward,done,_,_ = env.step(0)
        for i in range(n_frames):
            self.addObservation(observation)
            self.addNextObservation(observation)
        self.store(self.getState(),0,reward,done,self.getNextState())

    def sample_batch(self, batch_size=32):
        transitions = random.sample(self.buffer,batch_size) #that's a list
        batch = self.transition(*zip(*transitions)) #that's a gigantic transition in which every element is actually a list
        return batch

    def preprocessing(self, observation):
        '''
        Input:
            a frame, i.e. a (96,96,3)~(height,width,channels) tensor 
        Output:
            the same frame, but with shape (1,84,84) greyscale and normalized
        '''
        observation = observation.transpose(2,0,1) #Torch wants images in format (channels, height, width)
        observation = torch.from_numpy(observation).float()
        observation = self.rs(observation) # resize
        observation = self.gs(observation) # grayscale
        return (observation/255) # normalize
    
    def addObservation(self,observation):
        self.state.append(self.preprocessing(observation))
    
    def addNextObservation(self,next_observation):
        self.next_state.append(self.preprocessing(next_observation))

    def getState(self):
        state = torch.stack([observation for observation in self.state],0).squeeze()
        return state
    
    def getNextState(self):
        next_state = torch.stack([observation for observation in self.next_state],0).squeeze()
        return next_state

    def store(self,state,action,reward,done,next_state):
        '''
        Add an experience
        :param args: 
            observation(4frame),action,reward,done,next_observation(4frame)
        '''
        # self.addNextObservation(next_observation)
        # # print("Single observation"); print(self.state[-1].shape)#torch.Size([1, 84, 84])
        # state = torch.stack([observation for observation in self.state],0).squeeze()
        # # print("Stacked observations"); print(state.shape)#torch.Size([4, 84, 84])
        # next_state = torch.stack([observation for observation in self.next_state],0).squeeze()
        self.buffer.append(self.transition(state,action,reward,done,next_state))

    def burn_in_capacity(self):
        return len(self.buffer) / self.burn_in

    def capacity(self):
        return len(self.buffer) / self.memory_size

    def __iter__(self):
       return iter(self.buffer)
    
    def __len__(self):
        return len(self.buffer)