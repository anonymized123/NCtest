import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TAGConv


class Mutation_GAT(torch.nn.Module):
    def __init__(self, features, hidden, classes, dic):
        super(Mutation_GAT, self).__init__()
        self.gat1 = GATConv(features, hidden, **dic)
        self.gat2 = GATConv(hidden*dic['heads'], classes)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=4):
        super(GAT, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=4)
        self.gat2 = GATConv(hidden*heads, classes)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def get_Layer1(self, x, edge_index):
        x = self.gat1(x, edge_index)
        # x = F.relu(x)
        return x
    
    def get_Layer2(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        return x
    


class Mutation_GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, dic):
        super(Mutation_GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels, **dic)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def get_Layer1(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        return x
    
    def get_Layer2(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
    

class Mutation_GraphSAGE(torch.nn.Module):
    def __init__(self, feature, hidden, classes, dic):
        super(Mutation_GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(feature, hidden, **dic)
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def get_Layer1(self, x, edge_index):
        x = self.sage1(x, edge_index)
        # x = F.relu(x)
        return x
    
    def get_Layer2(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = self.sage2(x, edge_index)
        return x


class Mutation_TAGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dic):
        super(Mutation_TAGCN, self).__init__()
        self.conv1 = TAGConv(num_features, hidden_channels, **dic)
        self.conv2 = TAGConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class TAGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(TAGCN, self).__init__()
        self.conv1 = TAGConv(num_features, hidden_channels)
        self.conv2 = TAGConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_Layer1(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        return x
    
    def get_Layer2(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

