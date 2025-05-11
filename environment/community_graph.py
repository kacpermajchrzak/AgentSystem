import networkx as nx

def generate_graph(n_agents, m_=3):
    # ----- Parameters -----
    N = n_agents           # total number of nodes
    m = m_             # number of communities
    target_intra_deg = 10.0   # desired average intra-community degree
    d = 0.05            # fraction for inter-community connection probability

    block_size = N // m  # 50 nodes in each of 20 blocks
    # Probability of edges within a block to achieve average degree ~4
    p_intra = target_intra_deg / (block_size - 1)  

    # Probability of edges between blocks is d times p_intra
    p_inter = d * p_intra

    # Construct the probability matrix for the SBM:
    # - p_intra on the diagonal
    # - p_inter elsewhere
    p_matrix = [[p_inter]*m for _ in range(m)]
    for i in range(m):
        p_matrix[i][i] = p_intra

    # Block sizes: 20 blocks, each of size 50
    block_sizes = [block_size]*m

    # Generate the SBM graph
    G = nx.stochastic_block_model(block_sizes, p_matrix, seed=42)
    return G

