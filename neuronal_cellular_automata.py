import numpy as np
from matplotlib import pyplot as plt
from random import choice
from collections import Counter
from typing import List

class Cell_Phase:
    REST = 0
    FIRE = 1
    HYPER = 2
    REFRACT = 3

    PHASE_TRANSITION = {
        0 : lambda n, Tr: 1 if Cell_Phase.exceed_tr(n, Tr) else 0,
        1 : lambda n, Tr: 2,
        2 : lambda n, Tr: 3,
        3 : lambda n, Tr: 1 if Cell_Phase.exceed_tr(n, Tr) else 0 ,
    }

    def exceed_tr( neighbors, tr:float)-> bool:
        neighbor_type_count = Counter([c.type for c in neighbors])
        n_hyp = list(filter(lambda c: c.state==Cell_Phase.HYPER, neighbors))

        n_ex  = neighbor_type_count['E']
        n_in  = neighbor_type_count['I']
        return (n_ex - n_in - len(n_hyp)) >= tr


class Neuron:
    def __init__(self) -> None:
        self.AP = [np.random.rand() for _ in range(10)]
        self.state = choice([Cell_Phase.HYPER, Cell_Phase.REST, 
                             Cell_Phase.FIRE, Cell_Phase.REFRACT])
        self.type : str ['E' | 'I'] = 'E'
        self.neighbors : List[Neuron] = None

    def set_neighbors(self, neighbors) -> None:
        self.neighbors = neighbors
    
    def set_inhibitory(self):
        self.type = 'I'


class Neuronal_Lattice:

    def __init__(self, lattice_size:int, T_rest:float=8, T_rel:float=12, N_min:int=14, N_Max:int=60, percent_inhibit:float = 0.2):
        '''
        Initialize our CA
        '''
        self.T_rest = T_rest
        self.T_rel = T_rel
        self.N_min = N_min
        self.N_max = N_Max
        self.perc_i = percent_inhibit
        self.init_lattice(lattice_size)
        self.pt = Cell_Phase().PHASE_TRANSITION

    def init_lattice(self, size:int):
        '''
        Generate the lattice on which our CA will operate.
        '''
        if size**2 < self.N_max: 
            raise ValueError('Total # of cells in lattice must be larger than N_Max')
        
        self.lattice = [[ Neuron() for _ in range (size)] for _ in range (size)]
        cells = sum(self.lattice, [])
        for cell in cells:
            self.assign_neighbors(cell)

        num_inhibit = int(self.perc_i*(size**2))
        for inhibitory in np.random.choice(cells, num_inhibit):
            inhibitory.set_inhibitory()

        return self.lattice
    
    def assign_neighbors(self, cell:Neuron) -> None:
        '''
        Set each cell's neighbors to be a random distribution with 
        N_min < #_neighbors < N_max .

        * **cell** Neuron: The cell we are setting the neighbors for
        '''

        N = np.random.randint(self.N_min, self.N_max)
        neighbors = []
        for _ in range(N):
            rand_row = choice(self.lattice)
            rand_cell = choice(rand_row)
            if rand_cell not in neighbors and rand_cell != cell:
                neighbors.append(rand_cell)
        cell.set_neighbors(neighbors)
    
    def update_lattice(self) -> None:
        '''
        Iterate over all cells and calculate their next states based on their neighbors.
        ''' 
        [[self.next_state(cell) for cell in row] for row in self.lattice]

    def next_state(self, cell:Neuron) -> None:
        '''
        Calculate next state for a neuron given the cell and it's neighbors.

        * **cell** Neuron: the cell for which we are calculating the next state.
        '''
        

        tr = self.T_rest if cell.state == 0 else self.T_rel
        cell.state = self.pt[cell.state](cell.neighbors, tr)
        
    def show(self) -> None:
        '''
        Just a helper method to SEE what's going on in the cells.
        '''
        state_view = [[cell.state for cell in row] for row in self.lattice]
        plt.matshow(state_view)
        plt.show()

#%% Entry Point
if __name__ == '__main__':
    # generate lattice
    SIZE = 20
    NL = Neuronal_Lattice(SIZE)
    NL.show()
    
    ITERATIONS = 1000

    for _ in range(ITERATIONS):
        NL.update_lattice()
        if _ % 100 ==0:
            NL.show()
            