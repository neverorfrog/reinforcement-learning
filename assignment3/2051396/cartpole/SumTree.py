import math

class SumTree:
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.tree = [0] * (2 * capacity - 1) #nodes of the whole tree
        self.priorities = [None] * capacity #information lies on the leaves
        self.head = 0 #next index where i insert an element (treeIndex)
        self.size = 0 #number of nodes in the tree
        
    def get(self,sample):
        '''
        Given a sample, i.e. a value between 0 and root
        Returns index of tree and of batch and the priority related to the sample
        '''
        treeIndex = 0
        level = 0
        while (treeIndex*2 + 2) < self.size:
            left = 2*treeIndex + 1
            right = 2*treeIndex + 2
            level += 1
            
            if sample < self.tree[left]:
                treeIndex = left
            else:
                sample -= self.tree[left]
                treeIndex = right
        
        batchIndex = self.tree2batch(treeIndex,level)
        priority = self.tree[treeIndex]
        
        return treeIndex,priority,batchIndex
    
    def batch2tree(self,index,level):
        return index + 2**level - 1   
    
    def tree2batch(self,index,level):
        return index - 2**level + 1    
        
    def add(self,priority):
        '''Adds a priority to the tree'''
        self.tree[self.head] = priority
        self.update(self.head, priority)
        self.head  = (self.head + 1) % self.capacity
        self.size += 1
        return
    
    def update(self,index,priority):
        level = math.ceil(math.log(index+1,2))
        self.priorities[self.tree2batch(index, level)]
        
        parent = math.floor((index - 1)/2)
        while parent >= 0:
            self.tree[parent] += priority
            parent = math.floor((parent - 1)/2)
        
        return
    
    def root(self):
        return self.tree[0]
        
    
        
        
        