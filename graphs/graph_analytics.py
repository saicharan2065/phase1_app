import networkx as nx

class GraphAnalyticsEngine:
    def __init__(self, graph):
        self.graph = graph

    def generate_metrics(self):
        if self.graph is None or self.graph.number_of_nodes() == 0:
            return []
            
        # Calculate centrality measures
        degree_cen = nx.degree_centrality(self.graph)
        
        # For large graphs, betweenness and closeness can be slow, using approximations if possible
        if self.graph.number_of_nodes() > 1000:
            between_cen = nx.betweenness_centrality(self.graph, k=100)
            close_cen = nx.closeness_centrality(self.graph)
        else:
            between_cen = nx.betweenness_centrality(self.graph)
            close_cen = nx.closeness_centrality(self.graph)
            
        try:
            eigen_cen = nx.eigenvector_centrality_numpy(self.graph)
        except:
            eigen_cen = {n: 0 for n in self.graph.nodes()}
            
        pagerank = nx.pagerank(self.graph)
        
        results = []
        for node in self.graph.nodes():
            if str(node).startswith("CUST_"): # Only rank customers
                score = (degree_cen.get(node, 0) + between_cen.get(node, 0) + pagerank.get(node, 0)) * 100
                results.append({
                    "Entity ID": node,
                    "Degree Centrality": round(degree_cen.get(node, 0), 4),
                    "Betweenness Centrality": round(between_cen.get(node, 0), 4),
                    "Closeness Centrality": round(close_cen.get(node, 0), 4),
                    "Eigenvector Centrality": round(eigen_cen.get(node, 0), 4),
                    "PageRank": round(pagerank.get(node, 0), 4),
                    "Graph Risk Score": min(round(score, 2), 100)
                })
                
        return sorted(results, key=lambda x: x["Graph Risk Score"], reverse=True)
