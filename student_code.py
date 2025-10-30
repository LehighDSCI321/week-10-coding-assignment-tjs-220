"""
VersatileDigraph class for representing and visualizing directed graphs.
Supports Graphviz and Bokeh visualizations, with robust error handling.
"""
from collections import deque


class VersatileDigraph:
    """A class to represent a directed graph with visualization and analysis tools."""

    def __init__(self):
        """init"""
        self.nodes = {}  # {node_name: node_value}
        self.edges = {}  # {(source, target): (label, weight)}

    def add_node(self, name, value=1):
        """Adds a node to the graph with a given name and value."""
        if not isinstance(name, str):
            raise TypeError("Node name must be a string.")
        if not isinstance(value, (int, float)):
            raise TypeError("Node value must be numeric.")
        if value < 0:
            raise ValueError("Node value must be non-negative.")
        self.nodes[name] = value

    def add_edge(self, source, target, label=None, weight=None, edge_weight=None):
        """Adds edge"""
        if source not in self.nodes or target not in self.nodes:
            raise KeyError(f"Both source and target must be valid nodes: {source}, {target}")

        # Handle backward compatibility - edge_weight parameter takes precedence
        if edge_weight is not None:
            weight = edge_weight

        if label is not None and not isinstance(label, str):
            raise TypeError("Edge label must be a string.")
        if weight is not None and not isinstance(weight, (int, float)):
            raise TypeError("Edge weight must be numeric.")
        if weight is not None and weight < 0:
            raise ValueError("Edge weight must be non-negative.")

        self.edges[(source, target)] = (label if label else "", weight if weight else 0)

    def predecessors(self, node):
        """predecessors"""
        if node not in self.nodes:
            raise KeyError(f"Node '{node}' not found in graph.")
        return [src for (src, tgt) in self.edges if tgt == node]

    def get_node_value(self, name):
        """get node value"""
        if name not in self.nodes:
            raise KeyError(f"Node '{name}' not found.")
        return self.nodes[name]

    def get_edge_weight(self, source, target):
        """get edge weight"""
        if (source, target) not in self.edges:
            raise KeyError(f"Edge from '{source}' to '{target}' not found.")
        return self.edges[(source, target)][1]

    def plot_graph(self):
        """plot graph"""
        # pylint: disable=import-outside-toplevel
        try:
            from graphviz import Digraph
        except ImportError as exc:
            raise ImportError(
                "The 'plot_graph' function requires the 'graphviz' library. "
                "Please install it by running: pip install graphviz"
            ) from exc

        dot = Digraph()
        for node, value in self.nodes.items():
            dot.node(node, f"{node}\n({value})")
        for (src, tgt), (label, weight) in self.edges.items():
            dot.edge(src, tgt, f"{label}: {weight}")

    def successors(self, node):
        """defines successors"""
        if node not in self.nodes:
            raise KeyError(f"Node '{node}' not found in graph.")
        return [tgt for (src, tgt) in self.edges if src == node]

    def get_nodes(self):
        """defines get_nodes"""
        return list(self.nodes.keys())


class SortableDigraph(VersatileDigraph):
    """Extends VersatileDigraph with topological sorting capability."""

    def top_sort(self):
        """Returns a topologically sorted list of nodes using Kahn's algorithm."""

        # Step 1: Compute in-degree for each node
        in_degree = {node: 0 for node in self.nodes}
        for (src, tgt) in self.edges:
            in_degree[tgt] += 1

        # Step 2: Initialize queue with nodes having zero in-degree
        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        sorted_list = []

        # Step 3: Process nodes
        while queue:
            node = queue.popleft()
            sorted_list.append(node)

            # Decrease in-degree of successors
            for (src, tgt) in self.edges:
                if src == node:
                    in_degree[tgt] -= 1
                    if in_degree[tgt] == 0:
                        queue.append(tgt)

        # Step 4: Check for cycles
        if len(sorted_list) != len(self.nodes):
            raise ValueError("Graph contains a cycle; topological sort not possible.")

        return sorted_list


class TraversableDigraph(SortableDigraph):
    """Extends SortableDigraph with DFS and BFS traversal methods."""

    def dfs(self, start, visited=None):
        """Depth-first traversal from start node (excluding start if isolated)."""
        if start not in self.nodes:
            raise KeyError(f"Start node '{start}' not found.")
        if visited is None:
            visited = set()

        # Determine valid starting points
        roots = [n for n in self.successors(start)]
        if not roots:
            return  # No traversal possible if start has no successors

        for neighbor in sorted(roots):
            if neighbor not in visited:
                visited.add(neighbor)
                yield neighbor
                yield from self.dfs(neighbor, visited)

    def bfs(self, start):
        """Breadth-first traversal from start node (excluding start if isolated)."""
        if start not in self.nodes:
            raise KeyError(f"Start node '{start}' not found.")
        visited = set()
        queue = deque(sorted(self.successors(start)))  # Start from successors

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                yield node
                for neighbor in sorted(self.successors(node)):
                    if neighbor not in visited:
                        queue.append(neighbor)


class DAG(TraversableDigraph):
    """Directed Acyclic Graph that prevents cycles when adding edges."""

    def add_edge(self, source, target, label=None, weight=None, edge_weight=None):
        """Adds an edge only if it does not create a cycle."""
        if source not in self.nodes or target not in self.nodes:
            raise KeyError(f"Both source and target must be valid nodes: {source}, {target}")

        # Check for potential cycle: is there a path from target back to source?
        for node in self.dfs(target):
            if node == source:
                raise ValueError(
                    f"Adding edge from '{source}' to '{target}' would create a cycle.")

        # Safe to add edge
        super().add_edge(source, target, label=label, weight=weight, edge_weight=edge_weight)
