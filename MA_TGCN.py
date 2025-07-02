import torch
from torch_geometric_temporal.nn.recurrent.temporalgcn import TGCN



class MATGCN(torch.nn.Module):
    r"""
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        periods (int): Number of time periods.
        improved (bool): Stronger self loops (default :obj:`False`).
        cached (bool): Caching the message weights (default :obj:`False`).
        add_self_loops (bool): Adding self-loops for smoothing (default :obj:`True`).
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            periods: int,
            num_heads: int,
            improved: bool = False,
            cached: bool = False,
            add_self_loops: bool = True
    ):
        super(MATGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.num_heads = num_heads
        self._setup_layers()

    def _setup_layers(self):
        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 修改为多头注意力
        self._attention = torch.nn.Parameter(torch.empty(self.num_heads, self.periods, device=device))
        torch.nn.init.uniform_(self._attention)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** (PyTorch Float Tensor): Node features for T time periods.
            * **edge_index** (PyTorch Long Tensor): Graph edge indices.
            * **edge_weight** (PyTorch Long Tensor, optional)*: Edge weight vector.
            * **H** (PyTorch Float Tensor, optional): Hidden state matrix for all nodes.

        Return types:
            * **H** (PyTorch Float Tensor): Hidden state matrix for all nodes.
        """
        H_accum = 0
        for head in range(self.num_heads):
            probs = torch.nn.functional.softmax(self._attention[head], dim=0)
            H_head_accum = 0
            for period in range(self.periods):
                H_head_accum += probs[period] * self._base_tgcn(
                    X[:, :, period], edge_index, edge_weight, H
                )
            H_accum += H_head_accum
        return H_accum / self.num_heads
