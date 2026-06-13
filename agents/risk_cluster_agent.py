import networkx as nx

class RiskClusterDetector:
    def __init__(self, graph):
        self.graph = graph

    def detect_clusters(self):
        components = list(nx.connected_components(self.graph))
        clusters = []
        
        for idx, comp in enumerate(components):
            if len(comp) > 1:
                # Basic heuristic
                risk_score = min(len(comp) * 15, 100)
                clusters.append({
                    "Cluster ID": f"Cluster_{idx+1}",
                    "Members": ", ".join([str(n) for n in comp][:5]) + ("..." if len(comp) > 5 else ""),
                    "Size": len(comp),
                    "Risk Score": risk_score,
                    "Reason": "Shared connections"
                })
        
        return sorted(clusters, key=lambda x: x["Risk Score"], reverse=True)
