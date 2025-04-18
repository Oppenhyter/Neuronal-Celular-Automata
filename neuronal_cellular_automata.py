import numpy as np
from matplotlib import pyplot as plt
from random import choice, randint
from typing import List
from  matplotlib.animation import FuncAnimation
from scipy import signal
from functools import partial

class Neuron:
    '''
    Class for defining the attributes and functions specific to the neuron.
    '''

    def __init__(self, po:List[float]=None, ap:int=None, cell_type:str=None, neighbors:List=None, loc=None) -> None:
        '''
        Init dunder method for initializing the neuron.

        * **po** List[float]: the phase output we can provide back into the Neuron for the next generation.
        * **ap** int: the action potential. Can feed back into the next gen if we want to.
        * **cell_type** str: Excited ('E') or Inhibitory ('I).
        * **neighbors** List[Neuron]: references to the neighbor neurons we may feed to this instance for next generation neuron.

        **returns** None: just initializes the lattice
        '''
        self.loc = loc if loc != None else None
        self.phase_output : List[int]= po if po != None else [-randint(20,30), -randint(20,30), -randint(0,10), -randint(40,50),  -randint(20,30)]
        for _ in range (6): self.phase_output.append( self.phase_output[-1] ) 
        self.AP : int = ap if ap != None else randint(0,10)
        self.state : float = self.phase_output[self.AP]
        self.type : str ['E' | 'I'] = cell_type if cell_type != None else 'E'
        self.neighbors : List[Neuron] = neighbors

class Neuronal_Lattice:

    def __init__(self, lattice_size:int, T_rest:float=5, T_rel:float=10, N_min:int=16, N_max:int=60, 
                 a:float=0.6, percent_inhibit:float=0.2, neighbors:str ='far'):
        '''
        Initialize our Cellular Automata consisting of `Neurons`.

        * **lattice_size** int: the length and width of our square lattice
        * **T_rest** float: *(negative)*, the resting threshold required to activate the neuron
        * **T_rel** float: *(negative)*, the relative threshold required to return a neuron to a firing state while in refactory phase.
        * **N_min** int: the minimum number of neighbor neurons a cell may have.
        * **N_max** int: the maximum number of neighbor neurons a cell may have.
        * **a** float: the Hyperpolarization coefficient.
        * **percent_inhibit** float: the percentage of inhibitory neurons in the lattice.
        * **neighbors** str: close or far neighbors
        '''
        self.size :int = lattice_size
        self.T_rest: float = T_rest
        self.T_rel : float= T_rel
        self.N_min : int = N_min
        self.N_max : int = N_max
        self.a : float = a
        self.perc_i : float = percent_inhibit
        self.init_lattice(lattice_size, neighbors)
        self.time_series: List[float] = []

        self.lattice_wide = lambda f: [[f(cell) for cell in row] for row in self.lattice]

    def init_lattice(self, size:int, neighbors:str='far'):
        '''
        Generate the lattice on which our CA will operate.
        * **size** int: the length and width of our square lattice.
        * **neighbors** str: close or far neighbors
        '''
        if size**2 < self.N_max: 
            raise ValueError('Total # of cells in lattice must be larger than N_Max')
        
        self.lattice = [[ Neuron(loc = (x,y)) for x in range(size)] for y in range(size)]
        cells = sum(self.lattice, [])

        for cell in cells:
            if neighbors == 'far':
                self.assign_neighbors_far(cell)
            else: 
                self.assign_neighbors_close(cell)
        num_inhibit = int(self.perc_i*(size**2))

        for inhibitory in np.random.choice(cells, num_inhibit):
            inhibitory.type = 'I'
        return self.lattice

    def assign_neighbors_close(self, cell:Neuron) -> None:
        '''
        In the event you would like to see this not work as intended
        '''
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        in_range = lambda x, y:  x>=0 and x <=self.size-1 and y >= 0 and y <=self.size-1
        valid = list(filter(lambda v: in_range(v[0], v[1]), [np.array(cell.loc) + np.array(d) for d in directions]))
        cell.neighbors = [self.lattice[v[1]][v[0]] for v in valid]

    def assign_neighbors_far(self, cell:Neuron) -> None:
        '''
        Set each cell's neighbors to be a random distribution with 
        N_min < #_neighbors < N_max .

        * **cell** Neuron: The cell we are setting the neighbors for
        '''

        N = randint(self.N_min, self.N_max)
        neighbors = []
        while len(neighbors) <= N:
            rand_row = choice(self.lattice)
            rand_cell = choice(rand_row)
            if rand_cell not in neighbors and rand_cell != cell:
                neighbors.append(rand_cell)
        cell.neighbors = neighbors
    

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
        n_cell = Neuron(ap=cell.AP, cell_type=cell.type, neighbors=cell.neighbors)
        

        if cell.AP in (1,2,3,4):
            n_cell.AP = cell.AP+1
        else:
            n_cell.AP = self.transition(cell.AP, cell.neighbors, self.a)
        
        n_cell.state = n_cell.phase_output[n_cell.AP]
        return n_cell
        
    def states(self) -> None:
        '''
        Just a helper method to SEE what's going on in the cells.
        '''
        state_view = [[cell.state for cell in row] for row in self.lattice]
        return state_view

    def add_y(self, t):
        cells = sum(self.lattice, [])
        y = sum([cell.state  for cell in cells])

        self.time_series.append(y)


    def transition(self, AP:int, n:List, a:float):
        '''
        Helper method checking the 'rules' as stated in the paper.

        * **AP** int: the current action potential for the cell.
        * **n** List[Neuron]: references to this cell's neighbors.
        * **Tr** float: The threshold to overcome in order to become firing.
                Will refer to either T_rest (if resting) or T_rel (if refactory)
        * **a** float: the Hyper Coefficient.
        '''
       
        mode = 'rest' if AP==0 else 'ref'

        if self.exceed_tr(n, mode, a):
            if mode == 'rest':
                return 1
            elif mode == 'ref':
                return 0 if AP == 10 else AP+1
        else:
            return AP
        
    def exceed_tr(self, neighbors:List, mode:str, a:float)-> bool:
        '''
        The bulk of the 'rules' outlined in the paper.
        '''
        tr = self.T_rel if mode =='rel' else self.T_rest
        n_ex = sum([1 for c in neighbors if c.type == 'E' and c.AP in (1, 2, 3, 4)])
        n_in = sum([1 for c in neighbors if c.type == 'I' and c.AP in (1, 2, 3, 4)])
        n_hyp = sum([1 for c in neighbors if c.AP == 5])
        return (n_ex - n_in - a*n_hyp) >= tr


#%% Entry Point
if __name__ == '__main__':
    SIZE = 40
    
    #Base Case
    NL = Neuronal_Lattice(SIZE, a = -2)

    
    ITERATIONS = 750

    fig , ax = plt.subplots()
    init_states = NL.states()
    img = ax.imshow(init_states, cmap='gray', interpolation='antialiased')
    
    def update(t):
        NL.update_lattice()
        mat = NL.states()
        img.set_array(mat)

        NL.add_y(t)
        return [img]
    
    animation = FuncAnimation(fig, update, frames=ITERATIONS, repeat=False, interval=150, blit=True, save_count=300)
    plt.title("Neuronal Cellular Automata")
    
    plt.show()

    fig_xy  = plt.figure(figsize=(10,5))
    ax_xy = fig_xy.add_subplot(111)
    ax_xy.get_autoscalex_on()
    ax_xy.get_autoscaley_on()
    ax_xy.set_xlabel('x', fontsize=15)
    ax_xy.set_ylabel('y', fontsize=15)
    ax_xy.set_title(f'Neuron Activity', fontsize=25)
    ax_xy.plot(NL.time_series, 'b-', lw=0.5)
    plt.show()

    #pprint(NL.time_series)
    f, Pxx_den = signal.welch(NL.time_series, window='hann',fs=60)
   

    #f, Pxx_den = signal.periodogram(NL.time_series,fs=60 )
    
    plt.semilogy(f, Pxx_den)
    
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Power Spectrum Density (PSD)')
    plt.show()