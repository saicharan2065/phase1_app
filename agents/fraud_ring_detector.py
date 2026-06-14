import networkx as nx

class FraudRingDetector:
    def __init__(self, graph):
        self.graph = graph

    def detect_rings(self):
        if self.graph.number_of_nodes() == 0:
            return []
            
        components = list(nx.connected_components(self.graph))
        clusters = []
        
        # Calculate Graph Mathematics (PageRank & Centrality) to simulate GNN embeddings
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85)
            betweenness = nx.betweenness_centrality(self.graph)
        except:
            pagerank = {}
            betweenness = {}
        
        for idx, comp in enumerate(components):
            if len(comp) > 2: # At least 3 nodes to be considered a ring
                # Find how many are suspects vs shared resources
                suspects = [n for n in comp if str(n).startswith("SUSPECT_")]
                shared_resources = [n for n in comp if not str(n).startswith("SUSPECT_")]
                
                if len(suspects) > 1 and len(shared_resources) > 0:
                    # GNN Scoring: Aggregate PageRank and Betweenness
                    hub_score = 0
                    hub_name = "None"
                    for r in shared_resources:
                        score = (pagerank.get(r, 0) * 100) + (betweenness.get(r, 0) * 100)
                        if score > hub_score:
                            hub_score = score
                            hub_name = str(r)
                            
                    base_risk = min(len(suspects) * 20 + len(shared_resources) * 10, 80)
                    final_risk_score = min(base_risk + int(hub_score * 50), 100) # Boost by hub importance
                    
                    # Determine reason based on shared resources
                    reasons = []
                    if any(str(r).startswith("PHONE_") for r in shared_resources): reasons.append("Shared Phone")
                    if any(str(r).startswith("ADDR_") for r in shared_resources): reasons.append("Shared Address")
                    if any(str(r).startswith("EMAIL_") for r in shared_resources): reasons.append("Shared Email")
                    
                    clusters.append({
                        "Cluster ID": f"RING_{idx+1}",
                        "Members": len(comp),
                        "Suspects Involved": len(suspects),
                        "Hidden Hub Detected": hub_name,
                        "Hub Centrality Score": round(hub_score, 4),
                        "Risk Score": final_risk_score,
                        "Reason": ", ".join(reasons) if reasons else "Shared Graph Component"
                    })
        
        return sorted(clusters, key=lambda x: x["Risk Score"], reverse=True)
