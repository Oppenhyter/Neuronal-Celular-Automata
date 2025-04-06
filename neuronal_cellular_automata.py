import numpy as np
from pprint import pprint
from matplotlib import pyplot

class Cell_phases:
    def __init__(self):
        self.PHASE_TRANSITIONS = {
        'REST' : lambda n, Tr: 'FIRE' if self.exceed_tr(n, Tr) else 'REST',
        'FIRE' : lambda n, Tr: 'HYPER',
        'HYPER' : lambda n, Tr: 'REFRACT',
        'REFRACT' : lambda n, Tr: 'FIRE' if self.exceed_tr(n, Tr) else 'REST' ,
        }

    def exceed_tr(self, neighbors:list, tr)-> bool:
        n_ex  = neighbors['excited'] 
        n_in  = neighbors['inhibit']
        n_hyp = neighbors['hyper']
        return (len(n_ex) - len(n_in) - len(n_hyp)) >= tr

class Neuron:
    def __init__(self):
        self.AP = [np.random.rand() for _ in range(10)]
        self.state = None
        self.neighbors = None


class Neuronal_Lattice:

    def __init__(self, lattice_size:int, T_rest:float=8, T_rel:float=12, N_min:int=14, N_Max:int=60):
        '''
        Initialize our CA
        '''
        self.T_rest = T_rest
        self.T_rel = T_rel
        self.N_min = N_min
        self.N_max = N_Max
        self.lattice = self.init_lattice(lattice_size)
        self.pt = Cell_phases.PHASE_TRANSITIONS

    def init_lattice(self, size:int):
        '''
        Generate the lattice on which our CA will operate.
        '''
        if size < self.N_max: 
            raise ValueError('Total # of cells in lattice must be larger than N_Max')
            
        lattice = [[ Neuron() for _ in range (size) ] for _ in range (size)]
        #TODO set each cell's neighbors to be a random distribution.
        return lattice
    
    def update_lattice(self):
        '''
        '''
        raise NotImplementedError
    
    def next_state(self, cell, neighbors):
        '''
        '''
        self.pt[cell]
        raise NotImplementedError


#%% Entry Point
if __name__ == '__main__':
    # generate lattice
    SIZE = 2
    NL = Neuronal_Lattice(SIZE)

    #pyplot.matshow(CAL.lattice)
    pyplot.show()
    
