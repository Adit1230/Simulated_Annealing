import numpy as np
import random
import Simulated_annealing

class TSP(Simulated_annealing.State):
    def __init__(self, graph : np.ndarray):
        self.graph = graph
        self.path = []
        self.n_nodes = graph.shape()[0]

        unexplored = list(range(self.n_nodes))
        curr = 0

        self.path.append(curr)
        unexplored.remove(curr)

        for i in range(self.n_nodes):
            curr = unexplored[np.argmin(self.graph[curr, unexplored])]
            self.path.append(curr)
            unexplored.remove(curr)
        
    def get_neighbour(self):
        i = random.randrange(0, self.n_nodes)
        j = i
        while (j == i):
            j = random.randrange(0, self.n_nodes)
        
        i, j = min(i, j), max(i, j)

        return i*self.n_nodes + j, 0
    
    def cost(self, state):
        cost = 0
        for i in range(0, self.n_nodes):
            cost += self.graph[i, (i+1) % self.n_nodes]
        
        return cost
    
    def cost_change(self, idx, change):
        i = idx // self.n_nodes
        j = idx % self.n_nodes

        old_cost = self.graph[i, i+1] + self[j, (j+1) % self.n_nodes]
        new_cost = self.graph[i, (j+1) % self.n_nodes] + self.graph[j, i+1]

        return new_cost - old_cost

    def update(self, idx, change):
        i = idx // self.n_nodes
        j = idx % self.n_nodes

        self.path = self.path[0 : i] + self.path[i : j].reverse() + self.path[j:]


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML # Required for animation in Jupyter

# --- 1. Graph Generation Function ---

def generate_random_graph(n_nodes, max_distance=10):
    """
    Generates a random, symmetric adjacency matrix for a complete graph.

    Args:
        n_nodes (int): The number of nodes (cities).
        max_distance (int): The maximum distance between any two nodes.

    Returns:
        np.ndarray: The symmetric adjacency matrix (distance matrix).
    """
    if n_nodes < 2:
        return np.array([[0]])

    # Generate a random upper triangle (including the diagonal)
    # The distances are integers for simplicity
    np.random.seed(42) # Optional: set a seed for reproducibility
    upper_triangular = np.random.randint(1, max_distance + 1, size=(n_nodes, n_nodes))

    # Make it symmetric (A[i, j] = A[j, i])
    graph = np.triu(upper_triangular) + np.tril(upper_triangular.T, k=-1)

    # Set the diagonal (distance from a city to itself) to 0
    np.fill_diagonal(graph, 0)

    # Ensure all distances are non-negative
    graph = np.abs(graph)

    return graph

# --- 2. Path Visualization Function (for animation) ---

# Global variable to store node coordinates for consistent visualization
NODE_COORDINATES = None

def visualize_path_animation(path_history, graph_matrix, title="TSP Simulated Annealing"):
    """
    Visualizes the path history as an animation of a changing path over a graph.
    Designed to be run in a Jupyter Notebook.

    Args:
        path_history (list of list of int): A list where each element is a path 
                                            (list of node indices) at a certain step.
        graph_matrix (np.ndarray): The adjacency matrix used to solve the TSP.
        title (str): Title for the plot.
    """
    n_nodes = graph_matrix.shape[0]
    global NODE_COORDINATES

    # --- Setup Coordinates (Keep them consistent across all frames) ---
    if NODE_COORDINATES is None or NODE_COORDINATES.shape[0] != n_nodes:
        # Generate random coordinates for visualization
        # We use a circle layout for better visual separation
        theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        NODE_COORDINATES = np.column_stack((x, y))

    coords = NODE_COORDINATES
    
    # --- Matplotlib Setup ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')

    # Plot all nodes
    ax.scatter(coords[:, 0], coords[:, 1], s=200, c='blue', zorder=5) 
    
    # Label nodes
    for i in range(n_nodes):
        ax.text(coords[i, 0] + 0.05, coords[i, 1] + 0.05, str(i), fontsize=12, 
                ha='center', va='center')

    # Initial plot of the path (Path lines will be updated in the animation)
    line, = ax.plot([], [], 'r-', lw=2, zorder=3)
    start_node, = ax.plot([], [], 'go', markersize=10, zorder=4) # Mark the start node

    # --- Animation Function ---
    def update(frame_index):
        current_path = path_history[frame_index]
        
        # Coordinates for the path (including closing the loop)
        path_coords = coords[current_path + [current_path[0]]]
        
        # Update path line
        line.set_data(path_coords[:, 0], path_coords[:, 1])
        
        # Mark the starting node
        start_node_coord = coords[current_path[0]]
        start_node.set_data(start_node_coord[0], start_node_coord[1])
        
        # Calculate current cost for the title
        cost = 0
        for i in range(n_nodes):
            # Remember to check your path representation, 
            # assuming current_path[i] to current_path[(i+1)%n_nodes]
            u = current_path[i]
            v = current_path[(i + 1) % n_nodes]
            cost += graph_matrix[u, v]
            
        ax.set_title(f"{title}\nStep: {frame_index}, Cost: {cost:.2f}")
        return line, start_node

    # Create the animation object
    # interval is the delay in ms between frames. frames is the number of steps.
    ani = FuncAnimation(fig, update, frames=len(path_history), 
                        interval=200, blit=True)

    # Display the animation in the notebook
    # HTML(ani.to_jshtml()) can also be used, but is often slower
    display(ani) 
    plt.close(fig) # Prevent the static plot from showing

    # If you are NOT in a Jupyter/IPython environment, use:
    # plt.show() 
    
    # --- Optional: Draw all possible edges faintly ---
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            x_vals = [coords[i, 0], coords[j, 0]]
            y_vals = [coords[i, 1], coords[j, 1]]
            ax.plot(x_vals, y_vals, 'k:', alpha=0.1, zorder=1)
            
    # Show the final static plot (useful if not in a notebook)
    # plt.show()



            