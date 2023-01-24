def shortest_path(self,from_,to_):
        self.update_components()
        D=nx.Graph()
        e=[]
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                x=self.nodes[i]
                y=self.nodes[j]
                if (x.node_id, y.node_id in self.G.edges) and not x.isTransfering and not y.isTransfering:
                    e.append((x.node_id,y.node_id,self.G.edges[x.node_id,y.node_id]['weight']))
        D.add_weighted_edges_from(e)
        p=nx.dijkstra_path(D,from_,to_)
        return p