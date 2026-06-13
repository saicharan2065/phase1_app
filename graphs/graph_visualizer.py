from pyvis.network import Network
import os

class GraphVisualizer:
    def __init__(self, graph):
        self.graph = graph

    def generate_html(self, output_path="graph.html"):
        if self.graph.number_of_nodes() == 0:
            return "<h3>No data to visualize</h3>"
            
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            color = "#97C2FC"
            if node_type == "Customer":
                color = "#FF9999"
            elif node_type == "Phone":
                color = "#99FF99"
            elif node_type == "Email":
                color = "#FFFF99"
            elif node_type == "Address":
                color = "#FFCC99"
                
            net.add_node(node, label=str(node), title=f"Type: {node_type}", color=color)
            
        for u, v, data in self.graph.edges(data=True):
            net.add_edge(u, v, title=data.get("relationship", "connected_to"))
            
        net.toggle_physics(True)
        net.save_graph(output_path)
        
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h3>Graph could not be generated</h3>"
