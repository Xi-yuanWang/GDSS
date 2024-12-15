from utils.graph_utils import mask_adjs, pow_tensor
from models.layers import MLP
import torch.nn as nn
import torch
from typing_extensions import Callable, Final
from PygHO.pygho.honn.Conv import NGNNConv, GNNAKConv, DSSGNNConv, SSWLConv, SUNConv, PPGNConv, I2Conv, IGN2Conv
from PygHO.pygho.backend.MaTensor import MaskedTensor

gnn_str = "PPGN"
def transfermlpparam(mlp: dict):
    mlp = mlp.copy()
    mlp.update({"tailact": True, "numlayer": 2, "norm": "ln"})
    return mlp

maconvdict = {
    "SSWL":
    lambda dim, mlp: SSWLConv(dim, dim, "sum", "DD", transfermlpparam(mlp)
                              ),
    "DSSGNN":
    lambda dim, mlp: DSSGNNConv(dim, dim, "sum", "sum", "mean",
                                "DD", transfermlpparam(mlp)),
    "GNNAK":
    lambda dim, mlp: GNNAKConv(dim, dim, "sum", "mean", "DD",
                               transfermlpparam(mlp), transfermlpparam(mlp)),
    "SUN":
    lambda dim, mlp: SUNConv(dim, dim, "sum", "mean", "DD",
                             transfermlpparam(mlp), transfermlpparam(mlp)),
    "NGNN":
    lambda dim, mlp: NGNNConv(dim, dim, "sum", "DD", transfermlpparam(mlp)
                              ),
    "PPGN":
    lambda dim, mlp: PPGNConv(dim, dim, "sum", "DD", transfermlpparam(mlp)),
    "2IGN":
    lambda dim, mlp: IGN2Conv(dim, dim, "sum", "D", transfermlpparam(mlp))
}

class MaModel(nn.Module):
    residual: Final[bool]
    def __init__(self,
                 convfn: Callable,
                 num_layer=6,
                 hiddim=128,
                 residual=True,
                 mlp: dict = {}):
        super().__init__()
        self.residual = residual
        self.subggnns = nn.ModuleList(
            [convfn(hiddim, mlp) for _ in range(num_layer)])

    def forward(self, A: MaskedTensor, X: MaskedTensor):
        '''
        TODO: !warning input must be coalesced
        '''
        for conv in self.subggnns:
            tX = conv.forward(A, X, {})
            if self.residual:
                X = X.add(tX, samesparse=True)
            else:
                X = tX
        return X

class MyScoreNetwork(nn.Module):

    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
                    c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):

        super().__init__()
        
        self.mask = torch.ones([max_node_num, max_node_num]) - torch.eye(max_node_num)
        self.mask.unsqueeze_(0)

        self.temb = nn.Sequential(nn.Linear(1, c_hid), nn.SiLU(inplace=True), nn.Linear(c_hid, c_hid), nn.SiLU(inplace=True))

        self.gnn = MaModel(maconvdict[gnn_str], num_layers, c_hid, residual=True)
        print(gnn_str)
        self.final = nn.Sequential(nn.Linear(c_hid, c_hid), nn.SiLU(inplace=True), nn.Linear(c_hid, c_hid), nn.SiLU(inplace=True), nn.Linear(c_hid, 1))


    def forward(self, x, adj, flags, t):
        '''
        x: (B, N, d)
        adj: (B, N, N)
        '''
        B, N = adj.shape[0], adj.shape[1]

        adj = adj.unsqueeze(-1) * self.temb(t.to(adj.device).reshape(-1, 1, 1, 1))

        X = MaskedTensor(adj,
                 torch.ones((B, N, N), device=x.device, dtype=x.dtype),
                 0.0,
                 True)
        A = X
        X: MaskedTensor = self.gnn(A, X)
        X = X.data

        X = X + X.transpose(1, 2)

        score = self.final(X).squeeze(-1)
        
        # original input

        self.mask = self.mask.to(score.device)
        score = score * self.mask

        score = mask_adjs(score, flags)

        return score