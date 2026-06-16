from pyvis.network import Network
import os

class GraphVisualizer:
    def __init__(self, graph):
        self.graph = graph

    def generate_html(self, output_path="graph.html"):
        if self.graph.number_of_nodes() == 0:
            return "<h3>No data to visualize</h3>"
            
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        
        degree_dict = dict(self.graph.degree())
        
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            color = "#97C2FC"
            
            node_degree = degree_dict.get(node, 0)
            size = 15
            border_width = 1
            border_color = "#2B7CE9"
            
            if node_degree > 0:
                size = 25 + (node_degree * 2)
                border_width = 3
                border_color = "red"
                
            if node_type == "Suspect":
                color = "#FF9999"
            elif node_type == "Phone":
                color = "#99FF99"
            elif node_type == "Email":
                color = "#FFFF99"
            elif node_type == "Address":
                color = "#FFCC99"
                
            net.add_node(node, label=str(node), title=f"Type: {node_type} (Edges: {node_degree})", color={"background": color, "border": border_color}, size=size, borderWidth=border_width)
            
        for u, v, data in self.graph.edges(data=True):
            net.add_edge(u, v, title=data.get("relationship", "connected_to"))
            
        # Use Force Atlas 2 for fast layout, but stop physics after stabilization to prevent page freezing
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based",
            "stabilization": {
              "enabled": true,
              "iterations": 1000,
              "updateInterval": 100,
              "onlyDynamicEdges": false,
              "fit": true
            }
          }
        }
        """)
        net.save_graph(output_path)
        
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                html_content = f.read()
                import html
                escaped_html = html.escape(html_content)
                return f'<iframe srcdoc="{escaped_html}" width="100%" height="600px" style="border:none;"></iframe>'
        return "<h3>Graph could not be generated</h3>"
