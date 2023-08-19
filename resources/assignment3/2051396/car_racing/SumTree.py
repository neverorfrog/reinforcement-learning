import math
import numpy as np

class SumTree:
    
    def __init__(self,n_priorities):
        self.capacity = (2 * n_priorities - 1)
        self.tree = [0] * (2 * n_priorities - 1) #nodes of the whole tree
        self.priorities = [0] * n_priorities  #for printing
        self.transitions = np.zeros(n_priorities, dtype=object)
        self.head = 0 #next index where i insert an element (treeIndex)
        self.size = 0 #number of nodes in the tree
        
    def get(self,sample):
        '''
        Given a sample, i.e. a value between 0 and root
        Returns index of tree and of batch and the priority related to the sample
        '''
        index = 0
        level = 0
        while (index*2 + 2) < self.size:
            left = 2*index + 1
            right = 2*index + 2
            level += 1
            if sample <= self.tree[left]:
                index = left
            else:
                sample -= self.tree[left]
                index = right

        batchIndex = self.tree2batch(index)
        priority = self.priorities[batchIndex]
        transition = self.transitions[batchIndex]

        return index , priority , transition
    
    def tree2batch(self,index):
        level = math.floor(math.log(index+1,2))
        return index - 2**level + 1
    
    def batch2tree(self,index,level):
        return index + 2**level - 1   
        
    def add(self,transition,priority):
        
        self.tree[self.head] = priority
        batchIndex = self.tree2batch(self.head)
        self.priorities[batchIndex] = priority
        self.transitions[batchIndex] = transition
        
        if self.head % 2 == 0: #I'm adding a priority as right child
            priority += self.tree[self.head - 1]
        
        if self.head > 0:
            parent = math.floor((self.head - 1)/2)
            self.update(parent,priority,isLeaf=False)
        
        self.head  = (self.head + 1) % self.capacity
        self.size  = min(self.size+1, self.capacity)
        return
    
    def update(self,index,priority,isLeaf):
        #Assigning new priority
        change = priority - self.tree[index]
        self.tree[index] = priority
        if isLeaf:
            batchIndex = self.tree2batch(index)
            self.priorities[batchIndex] = priority
        
        #Propagating new priority
        parent = math.floor((index - 1)/2)
        while parent >= 0:
            self.tree[parent] += change
            parent = math.floor((parent - 1)/2)

            
    @property
    def root(self):
        return self.tree[0] #sum of the leaves
    
    def __repr__(self):
        return f"SumTree(tree={self.tree.__repr__()}, priorities={self.priorities.__repr__()})"
    
        
        
        