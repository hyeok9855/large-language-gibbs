import numpy as np
import networkx as nx
import jax.numpy as jnp
from jax import jit, vmap
from tqdm import tqdm

from collections import defaultdict, deque
from structure_learning.dag_gflownet.utils.exhaustive import all_dags, NUM_DAGS
from structure_learning.utils.nx_utils import (
    get_children,
    graph_to_adjacency,
    valid_action_mask_from_adjacency,
)


def get_gflownet_cache(gflownet, params, nodelist, batch_size=1024, verbose=True):
    """Cache the results of the GFlowNet for all the states.

    This function caches the log-probabilities for all the actions and for
    all the states of the GFlowNet.

    Parameters
    ----------
    gflownet : DAGGFlowNet instance
        The GFlowNet class, containing the definition of the model.

    params : GFNParameters instance
        The parameters of the GFlowNet (i.e., the parameters of the model, as
        well as log(Z)).

    nodelist : list
        The list of nodes; this list is required to ensure consistent
        encoding of nodes in the rows and columns of the adjacency matrix.

    batch_size : int
        The batch-size of the inputs of the GFlowNet.

    Returns
    -------
    cache : dict of (frozenset, np.ndarray)
        The cache of log-probabilities returned by the GFlowNet. The keys of
        the cache are the graphs (encoded as a frozenset of their edges), and
        the corresponding value is an array of size `(num_variables ** 2 + 1,)`
        containing the log-probabilities of all the actions in that state
        (including the "stop" action, at the last index).
    """
    gfn_apply = jit(vmap(gflownet.model.apply, in_axes=(None, 0, 0)))
    cache = dict()
    # node2idx = {n: i for i, n in enumerate(nodelist)}
    num_nodes = len(nodelist)

    def _flush_gfn_batch(_edge_keys, _adjacencies, _masks, _pad_to_batch_size):
        n_real = len(_edge_keys)
        adj = jnp.asarray(np.stack(_adjacencies, axis=0).astype(np.float32))
        msk = jnp.asarray(np.stack(_masks, axis=0).astype(np.float32))
        if _pad_to_batch_size and n_real < batch_size:
            pad = batch_size - n_real
            zeros = jnp.zeros((pad, num_nodes, num_nodes), dtype=jnp.float32)
            adj = jnp.concatenate([adj, zeros], axis=0)
            msk = jnp.concatenate([msk, zeros], axis=0)
        log_pis = gfn_apply(params.online, adj, msk)
        log_pis = np.asarray(log_pis[:n_real])
        for key, log_pi in zip(_edge_keys, log_pis):
            cache[key] = log_pi
        _edge_keys.clear()
        _adjacencies.clear()
        _masks.clear()

    edge_keys, adjacencies, masks = [], [], []
    for graph in tqdm(
        all_dags(num_nodes, nodelist=nodelist),
        desc="[get_gflownet_cache]",
        total=NUM_DAGS[num_nodes],
        disable=(not verbose),
        dynamic_ncols=True,
    ):
        adjacency = graph_to_adjacency(graph, nodelist)
        mask = valid_action_mask_from_adjacency(adjacency)

        edge_keys.append(frozenset(graph.edges()))
        adjacencies.append(adjacency)
        masks.append(mask)

        if len(edge_keys) >= batch_size:
            _flush_gfn_batch(edge_keys, adjacencies, masks, False)

    if edge_keys:
        _flush_gfn_batch(edge_keys, adjacencies, masks, len(edge_keys) < batch_size)

    return cache


def push_source_flow_to_terminal_states(gfn_state_graph, source_state_graph):
    """Compute a hashable key for a graph.

    This function traverses the GFlowNet state-action space graph (DAG) in a
    topologically sorted order and "pushes" the log_flow from each node to
    its children according to the log_prob_action specified on the edges.
    The topological sort ensures that all the flow has "arrived" at a node
    before "moving" its flow to its children.

    Parameters
    ----------
    gfn_state_graph : nx.DiGraph instance
        The GFlowNet state-action space where each node represents one GFlowNet
        state and each edge represents one GFlowNet action.

    source_state_graph: nx.DiGraph instance
        The graph representing the source state.

    Returns
    -------
    gfn_state_graph : nx.DiGraph instance
        The GFlowNet state-action space but now each node has an attribute
        named log_flow, which is -np.inf for non-terminal states and
        the marginal log probability for the terminal states.
    """
    # Initialize log_flow to be -np.inf (flow = 0) for all nodes
    nx.set_node_attributes(gfn_state_graph, -np.inf, "log_flow")

    # Except initialize log_flow to be 0 (flow = 1) for source node
    source_node_key = frozenset(source_state_graph.edges)
    nx.set_node_attributes(gfn_state_graph, {source_node_key: 0}, "log_flow")

    # Push flow through sorted graph
    for state in nx.topological_sort(gfn_state_graph):
        current_node = gfn_state_graph.nodes[state]
        log_flow_incoming = current_node[
            "log_flow"
        ]  # log_flow is log probability of reaching this node starting from source node

        # Compute terminal_log_flow
        stop_action_log_flow = current_node[
            "stop_action_log_flow"
        ]  # probability of taking stop action from this node

        # terminal prob = incoming probability * p(stop action at this node)
        current_node["terminal_log_flow"] = log_flow_incoming + stop_action_log_flow

        # Push flow along edges to children
        edges = gfn_state_graph.edges(state, data=True)
        for _, child, edge_attr in edges:
            log_prob_action = edge_attr["log_prob_action"]
            existing_log_flow_child = gfn_state_graph.nodes[child]["log_flow"]
            updated_log_flow_child = np.logaddexp(
                existing_log_flow_child, log_flow_incoming + log_prob_action
            )
            nx.set_node_attributes(gfn_state_graph, {child: updated_log_flow_child}, "log_flow")

    return gfn_state_graph


def construct_state_dag_with_bfs(gflownet_cache, nodelist, source_graph=None):
    """Constructs the state-action space of the GFlowNet.

    This function performs Breadth-First Search on the GFlowNet state-action space
    starting from the source state, in order to construct a networkx.DiGraph object
    where each node is a GFlowNet state and each edge is labeled with the action
    and the log probability of taking that action. Each node is also labeled with
    the stop_action_log_flow which contains the probability of terminating at that state.

    Parameters
    ----------
    gflownet_cache :

    nodelist :

    source_graph : nx.DiGraph instance
        The graph representing the source state.

    Returns
    -------
    gfn_state_graph : nx.DiGraph instance
        The GFlowNet state-action space.

    source_graph : nx.DiGraph instance
        The graph representing the source state.
    """
    gfn_state_graph = nx.DiGraph()
    is_state_queued = defaultdict(bool)
    states_to_visit = deque()

    if source_graph is None:
        source_graph = nx.DiGraph()
        source_graph.add_nodes_from(nodelist)
    source_graph_key = frozenset(source_graph.edges)

    gfn_state_graph.add_node(source_graph_key, graph=source_graph)
    states_to_visit.append(source_graph)
    is_state_queued[source_graph_key] = True
    while len(states_to_visit) > 0:
        current_graph = states_to_visit.popleft()
        current_graph_key = frozenset(current_graph.edges)
        children = get_children(current_graph, gflownet_cache, nodelist)
        for child_graph, action, log_prob in children:
            if action is None:  # stop action
                # Encode the stop action as a node attribute
                gfn_state_graph.nodes[current_graph_key]["stop_action_log_flow"] = log_prob
            else:
                child_graph_key = frozenset(child_graph.edges)
                if child_graph_key not in gfn_state_graph:
                    gfn_state_graph.add_node(child_graph_key, graph=child_graph)
                gfn_state_graph.add_edge(
                    current_graph_key, child_graph_key, action=action, log_prob_action=log_prob
                )
                already_visited = is_state_queued[child_graph_key]
                if not already_visited:
                    states_to_visit.append(child_graph)
                    is_state_queued[child_graph_key] = True

    return gfn_state_graph, source_graph


def posterior_exact(gflownet, params, nodelist, batch_size=256):
    gfn_cache = get_gflownet_cache(gflownet, params, nodelist, batch_size)
    gfn_state_graph, source_state_graph = construct_state_dag_with_bfs(gfn_cache, nodelist)
    gfn_state_graph = push_source_flow_to_terminal_states(gfn_state_graph, source_state_graph)
    return gfn_state_graph
