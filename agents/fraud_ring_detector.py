import networkx as nx

class FraudRingDetector:
    def __init__(self, graph):
        self.graph = graph

    def detect_rings(self):
        # We will use Community Detection if possible, but Connected Components works well for shared nodes
        components = list(nx.connected_components(self.graph))
        clusters = []
        
        for idx, comp in enumerate(components):
            if len(comp) > 2: # At least 3 nodes to be considered a ring
                # Find how many are customers vs shared resources
                customers = [n for n in comp if str(n).startswith("CUST_")]
                shared_resources = [n for n in comp if not str(n).startswith("CUST_")]
                
                if len(customers) > 1 and len(shared_resources) > 0:
                    risk_score = min(len(customers) * 20 + len(shared_resources) * 10, 100)
                    
                    # Determine reason based on shared resources
                    reasons = []
                    if any(str(r).startswith("PHONE_") for r in shared_resources): reasons.append("Shared Phone")
                    if any(str(r).startswith("ADDR_") for r in shared_resources): reasons.append("Shared Address")
                    if any(str(r).startswith("EMAIL_") for r in shared_resources): reasons.append("Shared Email")
                    if any(str(r).startswith("DEVICE_") for r in shared_resources): reasons.append("Shared Device")
                    
                    clusters.append({
                        "Cluster ID": f"RING_{idx+1}",
                        "Members": len(comp),
                        "Customers Involved": len(customers),
                        "Risk Score": risk_score,
                        "Reason": ", ".join(reasons) if reasons else "Shared Graph Component"
                    })
        
        return sorted(clusters, key=lambda x: x["Risk Score"], reverse=True)
