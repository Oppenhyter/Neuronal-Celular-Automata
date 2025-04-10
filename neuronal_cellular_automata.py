import numpy as np
from matplotlib import pyplot as plt
from random import choice, randint
from collections import Counter
from typing import List, Tuple, Any
from  matplotlib.animation import FuncAnimation

from scipy import signal

class Phase:

    #    AP   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    

    def transition(AP, n ,Tr, a):
        if AP == 10: 
            next_AP = 0 if Phase.exceed_tr(n, Tr, a) else AP 
            return next_AP
        else:
            next_AP = AP+1 if Phase.exceed_tr(n, Tr, a) else AP 
            return next_AP
        
    def exceed_tr(neighbors:List, tr:float, a:float)-> bool:
        n_ex = sum([c.state for c in neighbors if c.type == 'E' and c.AP in (1, 2, 3, 4)])
        n_in = sum([c.state for c in neighbors if c.type == 'I' and c.AP in (1, 2, 3, 4)])
        n_hyp = sum([c.state for c in neighbors if c.AP == 5])
        return (n_ex - n_in - a*n_hyp) >= tr


class Neuron:
    def __init__(self, cell_type = None, neighbors = None) -> None:
        self.phase_output = [2, randint(0,10), randint(0,10), randint(0,10), 
                             randint(0,5)]
        for _ in range (6):
            self.phase_output.append( self.phase_output[-1] ) 
        self.AP = randint(0,10)
        self.state = self.phase_output[self.AP]
        self.type : str ['E' | 'I'] = cell_type if cell_type != None else 'E'
        self.neighbors : List[Neuron] = neighbors

    def set_neighbors(self, neighbors) -> None:
        self.neighbors = neighbors
    
    def set_inhibitory(self):
        self.type = 'I'
    
    def update_state(self):
        self.state = self.phase_output[self.AP]

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
        self.time_series: List[float] = []

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

        #num = len(cells)
        #x = Counter([c.phase_output[c.AP] for c in cells])
        #[print(f'{y}: {(x[y]/num)*100:.0f}%') for y in x]

        return self.lattice
    
    def assign_neighbors(self, cell:Neuron) -> None:
        '''
        Set each cell's neighbors to be a random distribution with 
        N_min < #_neighbors < N_max .

        * **cell** Neuron: The cell we are setting the neighbors for
        '''

        N = randint(self.N_min, self.N_max)
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
        n_cell = Neuron(cell.type, cell.neighbors)
        n_cell.AP = Phase.transition(cell.AP, cell.neighbors, tr, self.a)
        n_cell.update_state()
        return n_cell
        
    def states(self) -> None:
        '''
        Just a helper method to SEE what's going on in the cells.
        '''
        state_view = [[cell.state for cell in row] for row in self.lattice]
        return state_view

    def add_y(self, t):
        cells = sum(self.lattice, [])
        y = sum([cell.state for cell in cells])

        self.time_series.append(y)


#%% Entry Point
if __name__ == '__main__':
    SIZE = 40
    NL = Neuronal_Lattice(SIZE, N_min=3, N_Max=5, T_rel=2, T_rest=1, a=0.1)
    ITERATIONS = 100

    fig , ax = plt.subplots()
    init_states = NL.states()
    img = ax.imshow(init_states, cmap='gray', interpolation='antialiased')
    
    def update(t):
        NL.update_lattice()
        mat = NL.states()
        img.set_array(mat)

        NL.add_y(t)
        
        return [img]
    
    animation = FuncAnimation(fig, update, frames=ITERATIONS, repeat=False, interval=10, blit=True)
    plt.show()

    fig_xy  = plt.figure(figsize=(10,10))
    ax_xy = fig_xy.add_subplot(111)
    ax_xy.get_autoscalex_on()
    ax_xy.get_autoscaley_on()
    ax_xy.set_xlabel('x', fontsize=15)
    ax_xy.set_ylabel('y', fontsize=15)
    ax_xy.set_title(f'Goop', fontsize=25)
    ax_xy.plot(NL.time_series, 'b-', lw=0.5)
    plt.show()

    #pprint(NL.time_series)
    f, Pxx_den = signal.welch(NL.time_series, window='hann')
    plt.semilogy(f, Pxx_den)

    #plt.ylim([1, 1])
    
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()