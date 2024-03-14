import torch
import torch.nn as nn


class AverageMIL(nn.Module):

    def __init__(self, nb_layers_in, n_classes):
        super(AverageMIL, self).__init__()

        self.name = 'Average'
        self.nb_layers_in = nb_layers_in
        self.n_classes = n_classes

        self.L = 500

        self.feats_extractor = nn.Sequential(nn.Linear(self.nb_layers_in, self.L),
                                             nn.ReLU())

        self.classifier = nn.Sequential(nn.Linear(self.L, self.n_classes))

    def forward(self, data):
        res = []
        for x in data:
            x = x.cuda()
            x = self.feats_extractor(x)
            x = x.mean(axis=0).unsqueeze(0)
            res.append(self.classifier(x))
        return torch.cat(res)
