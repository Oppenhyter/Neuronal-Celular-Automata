import numpy as np
from matplotlib import pyplot as plt
from random import choice
from collections import Counter
from typing import List
from  matplotlib.animation import FuncAnimation


class Phase:
    REST = 0
    FIRE = 1
    HYPER = 2
    REFRACT = 3

    def transition(x, AP, n ,Tr, a):
        match AP:
            case 0:
                return AP+1 if Phase.exceed_tr(n, Tr, a) else AP
            case 1 | 2 | 3 | 4:
                return AP+1 if Phase.exceed_tr(n, Tr, a) else AP
            case 5:
                return AP+1 if Phase.exceed_tr(n, Tr, a) else AP
                #return AP
            case 6|7|8|9: 
                return AP+1 if Phase.exceed_tr(n, Tr, a) else AP
            case 10:
                return 0 if Phase.exceed_tr(n, Tr, a) else AP

    def exceed_tr(neighbors:List, tr:float, a:float)-> bool:
        neighbor_type_count = Counter([c.type for c in neighbors if c.state == Phase.FIRE])
        n_hyp = list(filter(lambda c: c.state==Phase.HYPER, neighbors))
        n_ex  = neighbor_type_count['E']
        n_in  = neighbor_type_count['I']
        return (n_ex - n_in - a*len(n_hyp)) >= tr

    def get_phase(AP):
        match AP:
            case 0 : return Phase.REST
            case 1 | 2 | 3 | 4: return Phase.FIRE
            case 5 : return Phase.HYPER
            case 6 | 7 | 8 | 9 | 10: return Phase.REFRACT

class Neuron:
    def __init__(self) -> None:
        self.AP = np.random.randint(0,10)
        self.state = choice([Phase.HYPER, Phase.REST, 
                             Phase.FIRE, Phase.REFRACT])
        self.type : str ['E' | 'I'] = 'E'
        self.neighbors : List[Neuron] = None

    def set_neighbors(self, neighbors) -> None:
        self.neighbors = neighbors
    
    def set_inhibitory(self):
        self.type = 'I'
    
    def update_state(self):
        if self.AP == 0:
            self.state = Phase.REST
            return
        if self.AP < 5:
            self.state = Phase.FIRE
            return
        if self.AP < 6:
            self.state = Phase.HYPER
            return
        if self.AP <=10:
            self.state = Phase.REFRACT
            return


class Neuronal_Lattice:

    def __init__(self, lattice_size:int, T_rest:float=8, T_rel:float=12, N_min:int=14, N_Max:int=61, a:float=0.2, percent_inhibit:float = 0.2):
        '''
        Initialize our CA
        '''
        self.T_rest = T_rest
        self.T_rel = T_rel
        self.N_min = N_min
        self.N_max = N_Max
        self.a = a
        self.perc_i = percent_inhibit
        self.init_lattice(lattice_size)

        self.lattice_wide = lambda f: [[f(cell) for cell in row] for row in self.lattice]

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

        num = len(cells)
        x = Counter([Phase.get_phase(c.AP) for c in cells])
        [print(f'{y}: {(x[y]/num)*100:.0f}%') for y in x]

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
        next_lattice = self.lattice_wide(self.next_state)
        self.lattice = next_lattice

    def next_state(self, cell:Neuron) -> None:
        '''
        Calculate next state for a neuron given the cell and it's neighbors.

        * **cell** Neuron: the cell for which we are calculating the next state.
        '''
        tr = self.T_rest if cell.state == 0 else self.T_rel
        n_cell = cell
        n_cell.AP = Phase().transition(cell.AP, cell.neighbors, tr, self.a)
        n_cell.update_state()
        return n_cell
        
    def states(self) -> None:
        '''
        Just a helper method to SEE what's going on in the cells.
        '''
        state_view = [[cell.state for cell in row] for row in self.lattice]
        return state_view

        


#%% Entry Point
if __name__ == '__main__':
    # generate lattice
    SIZE = 20
    NL = Neuronal_Lattice(SIZE, a=0.1, percent_inhibit=0.2)
    ITERATIONS = 100

    fig , ax = plt.subplots()
    init_states = NL.states()
    img = ax.imshow(init_states, cmap='gray', interpolation='nearest')
    
    def update(f):
        NL.update_lattice()
        mat = NL.states()
        img.set_array(mat)
        return [img]
    
    #[[print(cell.AP) for cell in row] for row in NL.lattice]
    animation = FuncAnimation(fig, update, frames=ITERATIONS,repeat=True, interval=10, blit=True)
    plt.show()
            