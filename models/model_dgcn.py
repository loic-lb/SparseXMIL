import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torch_geometric.nn import GINConv
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import RadiusGraph, KNNGraph
from torch_geometric.nn import aggr


class AttentionGate(nn.Module):
    def __init__(self, L=256, D=256, dropout=None, K=1):
        super(AttentionGate, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, K)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        c = self.attention_c(a * b)
        return c


######################################
# DeepGraphConv Implementation #
######################################
class DGCNMIL(nn.Module):
    def __init__(self, num_features=1024, hidden_dim=256, dropout=0.25, n_classes=2):
        super(DGCNMIL, self).__init__()
        self.name = "DGCNMIL"
        self.num_features = num_features
        self.n_classes = n_classes
        # Uncomment to use radius graph instead of knn
        #self.r = 0.5
        self.neigh = 8
        self.graph_transform = KNNGraph(k=self.neigh)

        self.conv1 = GINConv(Seq(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

        self.attention_gate = nn.Sequential(AttentionGate(L=hidden_dim, D=hidden_dim, dropout=dropout, K=1))

        self.path_attention_head = aggr.AttentionalAggregation(self.attention_gate)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def create_batch(self, data, tiles):
        """
        Creates a batch of graphs from extracted features and tiles coordinates
        @param data: list of extracted features
        @param tiles: list of tiles coordinates
        @return: a pytorch geometric batch of graphs
        """
        batch = []
        for i in range(len(data)):
            g = Data(x=data[i], pos=tiles[i])
            # Uncomment to use radius graph instead of knn
            #max_dist = torch.cdist(tiles[i].min(axis=0)[0][None, ].float(),
            #                       tiles[i].max(axis=0)[0][None, ].float())
            #graph_transform = RadiusGraph(r=self.r * max_dist)
            g = self.graph_transform(g)
            batch.append(g)
        return Batch.from_data_list(batch)

    def forward(self, x, tiles_locations, return_attention=False):

        x = self.create_batch(x, tiles_locations).cuda()

        x1 = F.relu(self.conv1(x=x.x, edge_index=x.edge_index))
        x2 = F.relu(self.conv2(x=x1, edge_index=x.edge_index))
        x3 = F.relu(self.conv3(x=x2, edge_index=x.edge_index))

        if return_attention:
            attention_weights = []
            for batch_idx in x.batch.unique():
                x_A = x3[x.batch == batch_idx, ]
                A = self.attention_gate(x_A)
                A = torch.transpose(A, 1, 0)
                attention_weights.append(F.softmax(A, dim=1))

        h_path = self.path_attention_head(x3, x.batch)
        h_path = self.path_rho(h_path)

        if return_attention:
            return self.classifier(h_path), attention_weights
        else:
            return self.classifier(h_path)
