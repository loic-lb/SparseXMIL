import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):

    def __init__(self, nb_layers_in, n_classes):
        super(GatedAttention, self).__init__()

        self.name = 'Attention'
        self.nb_layers_in = nb_layers_in
        self.L = 500
        self.D = 128
        self.K = 1
        self.n_classes = n_classes

        self.feats_extractor = nn.Sequential(nn.Linear(self.nb_layers_in, self.L),
                                             nn.ReLU())

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D),
                                         nn.Tanh())

        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D),
                                         nn.Sigmoid())

        self.attention_network = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, self.n_classes))

    def forward(self, data, return_attention=False):
        res = []
        if return_attention:
            attention_weights = []
        for x in data:
            x = x.cuda()
            x = self.feats_extractor(x)
            A_U = self.attention_U(x)
            A_V = self.attention_V(x)

            A = self.attention_network(A_V * A_U)
            A = torch.transpose(A, 1, 0)
            A = F.softmax(A, dim=1)

            if return_attention:
                attention_weights.append(A)
            M = torch.mm(A, x)

            res.append(self.classifier(M))

        if return_attention:
            return torch.cat(res), attention_weights
        else:
            return torch.cat(res)
