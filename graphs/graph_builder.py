import networkx as nx
import pandas as pd

class EntityGraphEngine:
    def __init__(self):
        self.graph = nx.Graph()

    def build_from_dataframe(self, df):
        self.graph.clear()
        if df is None or df.empty:
            return
            
        id_col = next((c for c in df.columns if 'id' in c.lower()), df.columns[0])
        phone_col = next((c for c in df.columns if 'phone' in c.lower() or 'mobile' in c.lower()), None)
        email_col = next((c for c in df.columns if 'email' in c.lower()), None)
        address_col = next((c for c in df.columns if 'address' in c.lower() or 'residence' in c.lower()), None)

        for _, row in df.iterrows():
            entity_id = f"CUST_{row[id_col]}"
            self.graph.add_node(entity_id, type="Customer")
            
            if phone_col and pd.notna(row[phone_col]):
                phone_node = f"PHONE_{row[phone_col]}"
                self.graph.add_node(phone_node, type="Phone")
                self.graph.add_edge(entity_id, phone_node, relationship="owns")
                
            if email_col and pd.notna(row[email_col]):
                email_node = f"EMAIL_{row[email_col]}"
                self.graph.add_node(email_node, type="Email")
                self.graph.add_edge(entity_id, email_node, relationship="owns")
                
            if address_col and pd.notna(row[address_col]):
                addr_node = f"ADDR_{row[address_col]}"
                self.graph.add_node(addr_node, type="Address")
                self.graph.add_edge(entity_id, addr_node, relationship="uses")
                
    def get_statistics(self):
        return {
            "Node Count": self.graph.number_of_nodes(),
            "Edge Count": self.graph.number_of_edges(),
            "Connected Components": nx.number_connected_components(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }
