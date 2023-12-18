from __future__ import division
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from collections import OrderedDict
import dgl
import networks.init as init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ModuleType = Type[nn.Module]


def miniblock(
    input_size: int,
    output_size: int = 0,
    norm_layer: Optional[ModuleType] = None,
    activation: Optional[ModuleType] = None,
    linear_layer: Type[nn.Linear] = nn.Linear,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and \
    activation."""
    layers: List[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        layers += [norm_layer(output_size)]  # type: ignore
    if activation is not None:
        layers += [activation()]
    return layers

def null_activation(x):
    return x


class MLP(nn.Module):
    """Simple MLP backbone.

    Create a MLP of size input_dim * hidden_sizes[0] * hidden_sizes[1] * ...
    * hidden_sizes[-1] * output_dim

    :param int input_dim: dimension of the input vector.
    :param int output_dim: dimension of the output vector. If set to 0, there
        is no final linear layer.
    :param hidden_sizes: shape of MLP passed in as a list, not including
        input_dim and output_dim.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: which device to create this model on. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        device: Optional[Union[str, int, torch.device]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_sizes))]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
            else:
                activation_list = [activation for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, norm, activ in zip(
            hidden_sizes[:-1], hidden_sizes[1:], norm_layer_list, activation_list
        ):
            model += miniblock(in_dim, out_dim, norm, activ, linear_layer)
        if output_dim > 0:
            model += [linear_layer(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim or hidden_sizes[-1]
        self.model = nn.Sequential(*model)

    def forward(self, s: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if self.device is not None:
            s = torch.as_tensor(
                s,
                device=self.device,  # type: ignore
                dtype=torch.float32,
            )
        return self.model(s.flatten(1))  # type: ignore


class SoftNet(nn.Module):
    def __init__(self, output_shape,
                 base_type, em_input_shape, input_shape,
                 em_hidden_shapes,
                 hidden_shapes,
                 num_layers, num_modules,
                 module_hidden,
                 gating_hidden, num_gating_layers,
                 add_bn=False,
                 pre_softmax=False,
                 dueling_param=None,
                 device='cuda',
                 cond_ob=True,
                 module_hidden_init_func=init.basic_init,
                 last_init_func=init.uniform_init,
                 activation_func=F.relu,
                 is_last=True,
                 softmax=False,
                 **kwargs):

        super().__init__()

        self.base = base_type(
                        last_activation_func = null_activation,
                        input_shape = input_shape,
                        activation_func = activation_func,
                        hidden_shapes = hidden_shapes,
                        **kwargs )
        self.em_base = base_type(
                        last_activation_func = null_activation,
                        input_shape = em_input_shape,
                        activation_func = activation_func,
                        hidden_shapes = em_hidden_shapes,
                        **kwargs )

        self.em_input_shape = em_input_shape
        self.input_shape = input_shape

        self.activation_func = activation_func

        module_input_shape = self.base.output_shape
        self.layer_modules = []

        self.num_layers = num_layers
        self.num_modules = num_modules

        self.device = device
        self.use_dueling = dueling_param is not None
        self.is_last = is_last
        self.softmax = softmax

        for i in range(num_layers):
            layer_module = []
            for j in range( num_modules ):
                fc = nn.Linear(module_input_shape, module_hidden)
                module_hidden_init_func(fc)
                if add_bn:
                    module = nn.Sequential(
                        nn.BatchNorm1d(module_input_shape),
                        fc,
                        nn.BatchNorm1d(module_hidden)
                    )
                else:
                    module = fc

                layer_module.append(module)
                self.__setattr__("module_{}_{}".format(i,j), module)

            module_input_shape = module_hidden
            self.layer_modules.append(layer_module)

        self.last = nn.Linear(module_input_shape, output_shape)
        last_init_func( self.last )

        assert self.em_base.output_shape == self.base.output_shape, \
            "embedding should has the same dimension with base output for gated"
        gating_input_shape = self.em_base.output_shape
        self.gating_fcs = []
        for i in range(num_gating_layers):
            gating_fc = nn.Linear(gating_input_shape, gating_hidden)
            module_hidden_init_func(gating_fc)
            self.gating_fcs.append(gating_fc)
            self.__setattr__("gating_fc_{}".format(i), gating_fc)
            gating_input_shape = gating_hidden

        self.gating_weight_fcs = []
        self.gating_weight_cond_fcs = []

        self.gating_weight_fc_0 = nn.Linear(gating_input_shape,
                                            num_modules * num_modules )
        last_init_func( self.gating_weight_fc_0)
        # self.gating_weight_fcs.append(self.gating_weight_fc_0)

        for layer_idx in range(num_layers-2):
            gating_weight_cond_fc = nn.Linear((layer_idx+1) *
                                              num_modules * num_modules,
                                              gating_input_shape)
            module_hidden_init_func(gating_weight_cond_fc)
            self.__setattr__("gating_weight_cond_fc_{}".format(layer_idx+1),
                             gating_weight_cond_fc)
            self.gating_weight_cond_fcs.append(gating_weight_cond_fc)

            gating_weight_fc = nn.Linear(gating_input_shape,
                                         num_modules * num_modules)
            last_init_func(gating_weight_fc)
            self.__setattr__("gating_weight_fc_{}".format(layer_idx+1),
                             gating_weight_fc)
            self.gating_weight_fcs.append(gating_weight_fc)

        self.gating_weight_cond_last = nn.Linear((num_layers-1) * \
                                                 num_modules * num_modules,
                                                 gating_input_shape)
        module_hidden_init_func(self.gating_weight_cond_last)

        self.gating_weight_last = nn.Linear(gating_input_shape, num_modules)
        last_init_func( self.gating_weight_last )

        self.pre_softmax = pre_softmax
        self.cond_ob = cond_ob

        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = output_shape, 1
            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": output_shape,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": output_shape,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(self, x, state=None, info=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        state_in, embedding_in = torch.split(x, [self.input_shape, self.em_input_shape], dim=1)
        out = self.base(state_in)
        embedding = self.em_base(embedding_in)

        if self.cond_ob:
            embedding = embedding * out

        out = self.activation_func(out)

        if len(self.gating_fcs) > 0:
            embedding = self.activation_func(embedding)
            for fc in self.gating_fcs[:-1]:
                embedding = fc(embedding)
                embedding = self.activation_func(embedding)
            embedding = self.gating_fcs[-1](embedding)

        base_shape = embedding.shape[:-1]

        weights = []
        flatten_weights = []

        raw_weight = self.gating_weight_fc_0(self.activation_func(embedding))

        weight_shape = base_shape + torch.Size([self.num_modules,
                                                self.num_modules])
        flatten_shape = base_shape + torch.Size([self.num_modules * \
                                                self.num_modules])

        raw_weight = raw_weight.view(weight_shape)

        softmax_weight = F.softmax(raw_weight, dim=-1)
        weights.append(softmax_weight)
        if self.pre_softmax:
            flatten_weights.append(raw_weight.view(flatten_shape))
        else:
            flatten_weights.append(softmax_weight.view(flatten_shape))

        for gating_weight_fc, gating_weight_cond_fc in zip(self.gating_weight_fcs, self.gating_weight_cond_fcs):
            cond = torch.cat(flatten_weights, dim=-1)
            if self.pre_softmax:
                cond = self.activation_func(cond)
            cond = gating_weight_cond_fc(cond)
            cond = cond * embedding
            cond = self.activation_func(cond)

            raw_weight = gating_weight_fc(cond)
            raw_weight = raw_weight.view(weight_shape)
            softmax_weight = F.softmax(raw_weight, dim=-1)
            weights.append(softmax_weight)
            if self.pre_softmax:
                flatten_weights.append(raw_weight.view(flatten_shape))
            else:
                flatten_weights.append(softmax_weight.view(flatten_shape))

        cond = torch.cat(flatten_weights, dim=-1)
        if self.pre_softmax:
            cond = self.activation_func(cond)
        cond = self.gating_weight_cond_last(cond)
        cond = cond * embedding
        cond = self.activation_func(cond)

        raw_last_weight = self.gating_weight_last(cond)
        last_weight = F.softmax(raw_last_weight, dim = -1)

        module_outputs = [(layer_module(out)).unsqueeze(-2) \
                for layer_module in self.layer_modules[0]]

        module_outputs = torch.cat(module_outputs, dim = -2 )


        for i in range(self.num_layers - 1):
            new_module_outputs = []
            for j, layer_module in enumerate(self.layer_modules[i + 1]):
                module_input = (module_outputs * \
                    weights[i][..., j, :].unsqueeze(-1)).sum(dim=-2)

                module_input = self.activation_func(module_input)
                new_module_outputs.append((
                        layer_module(module_input)
                ).unsqueeze(-2))

            module_outputs = torch.cat(new_module_outputs, dim = -2)

        out = (module_outputs * last_weight.unsqueeze(-1)).sum(-2)
        out = self.activation_func(out)
        if self.is_last:
            out = self.last(out)

        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(out), self.V(out)
            out = q - q.mean(dim=1, keepdim=True) + v
        if self.softmax:
            out = torch.softmax(out, dim=-1)
        return out, state


class GCNlayer(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 device='cpu'):
        super(GCNlayer, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.layers.append(GraphConv(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden))
        self.layers.append(GraphConv(n_hidden, n_classes))

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != self.n_layers:
                h = F.relu(h)
        return h


class SelfAttentionNet(nn.Module):
    def __init__(self, output_shape,
                 em_input_shape, state_input_shape,
                 task_num: int,
                 hidden_type='GCN', graph=None,
                 dueling_param=None, device='cuda',
                 ):

        super().__init__()

        self.em_input_shape = em_input_shape
        self.state_input_shape = state_input_shape
        self.output_shape = output_shape
        self.device = device
        self.g = graph
        self.use_dueling = dueling_param is not None
        self.task_num = task_num
        self.f = nn.Softmax(dim=2)
        self.is_gcn = False

        if hidden_type == 'MLP':
            self.w_k = nn.Sequential(OrderedDict([
                ('k1', nn.Linear(int(state_input_shape), 128, bias=False)),
                ('kRelu', nn.ReLU()),
                ('k2', nn.Linear(128, 64, bias=False))
            ]))

            self.w_v = nn.Sequential(OrderedDict([
                ('v1', nn.Linear(int(state_input_shape), 128, bias=False)),
                ('vRelu', nn.ReLU()),
                ('v2', nn.Linear(128, 64, bias=False))
            ]))
        elif hidden_type == 'GCN':
            self.is_gcn = True
            self.w_k = GCNlayer(in_feats=4, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            self.w_v = GCNlayer(in_feats=4, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
        else:
            raise Exception('Invalid layer type, must be either "GCN" or "MLP"')

        self.w_q = nn.Sequential(OrderedDict([
            ('q1', nn.Linear(em_input_shape, 128)),
            ('qRelu', nn.ReLU()),
            ('q2', nn.Linear(128, 64))
        ]))

        if self.use_dueling:  # dueling DQN
            Q_kwargs, V_kwargs = dueling_param
            Q_hidden_size = Q_kwargs['hidden_sizes'][0]
            V_hidden_size = V_kwargs['hidden_sizes'][0]
            self.Q = nn.Sequential(OrderedDict([
                ('Q1', nn.Linear(64, Q_hidden_size)),
                ('QRelu', nn.ReLU()),
                ('Q2', nn.Linear(Q_hidden_size, output_shape))
            ]))
            self.V = nn.Sequential(OrderedDict([
                ('v1', nn.Linear(64, V_hidden_size)),
                ('VRelu', nn.ReLU()),
                ('V2', nn.Linear(V_hidden_size, 1))
            ]))
        else:
            assert False, 'MAM must use dueling DQN architecture.'

    def forward(self, x, state=None, show_grid=False, info=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        state_in, embedding_in = torch.split(x, [self.state_input_shape, self.em_input_shape*self.task_num], dim=1)
        embedding_in = torch.reshape(embedding_in, [-1, self.task_num, self.em_input_shape])

        ba = state_in.shape[0]
        bus_num = int(self.state_input_shape/4)

        q = self.w_q(embedding_in)
        q = torch.mean(q, dim=1).reshape(-1, 1, 64)

        if self.is_gcn:
            bg = dgl.batch([self.g for _ in range(ba)]).to(self.device)
            state_in = state_in.reshape(-1, 4)
            k = self.w_k(bg, state_in).reshape(-1, bus_num, 64)
            v = self.w_v(bg, state_in).reshape(-1, bus_num, 64)
        else:
            k = self.w_k(state_in).reshape(ba, -1, 64)
            v = self.w_v(state_in).reshape(ba, -1, 64)

        temperature = k.shape[0] ** 0.5

        attn = torch.bmm(q, (k/temperature).transpose(1, 2))
        attn = self.f(attn)
        out = torch.bmm(attn, v).view(ba, -1)

        if self.use_dueling:  # Dueling DQN
            Q, V = self.Q(out), self.V(out)
            out = Q - Q.mean(dim=1, keepdim=True) + V
        else:
            assert False, 'MAM must use dueling DQN architecture.'
        if show_grid:
            return out, state, attn.detach()
        else:
            return out, state


class SelfAttentionNetWeighted(nn.Module):
    def __init__(self, output_shape,
                 em_input_shape, state_input_shape,
                 task_num: int,
                 hidden_type='GCN', graph=None,
                 dueling_param=None, device='cuda',
                 ):

        super().__init__()

        self.em_input_shape = em_input_shape
        self.state_input_shape = state_input_shape
        self.output_shape = output_shape
        self.device = device
        self.g = graph
        self.use_dueling = dueling_param is not None
        self.task_num = task_num
        self.f = nn.Softmax(dim=2)
        self.f0 = nn.Softmax(dim=0)
        self.is_gcn = False

        if hidden_type == 'MLP':
            self.w_k = nn.Sequential(OrderedDict([
                ('k1', nn.Linear(int(state_input_shape), 128, bias=False)),
                ('kRelu', nn.ReLU()),
                ('k2', nn.Linear(128, 64, bias=False))
            ]))

            self.w_v = nn.Sequential(OrderedDict([
                ('v1', nn.Linear(int(state_input_shape), 128, bias=False)),
                ('vRelu', nn.ReLU()),
                ('v2', nn.Linear(128, 64, bias=False))
            ]))
        elif hidden_type == 'GCN':
            self.is_gcn = True
            self.w_k = GCNlayer(in_feats=4, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
            self.w_v = GCNlayer(in_feats=4, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
        else:
            raise Exception('Invalid layer type, must be either "GCN" or "MLP"')

        self.w_q = nn.Sequential(OrderedDict([
            ('q1', nn.Linear(em_input_shape, 128)),
            ('qRelu', nn.ReLU()),
            ('q2', nn.Linear(128, 64))
        ]))

        self.alpha = nn.Parameter(torch.Tensor([1/self.task_num for _ in range(task_num)]), requires_grad=True)

        if self.use_dueling:  # dueling DQN
            Q_kwargs, V_kwargs = dueling_param

            Q_hidden_size = Q_kwargs['hidden_sizes'][0]
            V_hidden_size = V_kwargs['hidden_sizes'][0]
            print(Q_hidden_size)
            self.Q = nn.Sequential(OrderedDict([
                ('Q1', nn.Linear(64, Q_hidden_size)),
                ('QRelu', nn.ReLU()),
                ('Q2', nn.Linear(Q_hidden_size, output_shape))
            ]))
            self.V = nn.Sequential(OrderedDict([
                ('v1', nn.Linear(64, V_hidden_size)),
                ('VRelu', nn.ReLU()),
                ('V2', nn.Linear(V_hidden_size, 1))
            ]))
        else:
            assert False, 'MAM must use dueling DQN architecture.'

    def forward(self, x, state=None, show_grid=False, info=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        state_in, embedding_in = torch.split(x, [self.state_input_shape, self.em_input_shape*self.task_num], dim=1)
        embedding_in = torch.reshape(embedding_in, [-1, self.task_num, self.em_input_shape])

        ba = state_in.shape[0]
        bus_num = int(self.state_input_shape/4)

        q = self.w_q(embedding_in)
        self.alpha.data = self.f0(self.alpha)
        q = torch.matmul(self.alpha, q).reshape(ba, 1, 64)

        if self.is_gcn:
            bg = dgl.batch([self.g for _ in range(ba)]).to(self.device)
            state_in = state_in.reshape(-1, 4)
            k = self.w_k(bg, state_in).reshape(-1, bus_num, 64)
            v = self.w_v(bg, state_in).reshape(-1, bus_num, 64)
        else:
            k = self.w_k(state_in).reshape(ba, -1, 64)
            v = self.w_v(state_in).reshape(ba, -1, 64)

        temperature = k.shape[0] ** 0.5

        attn = torch.bmm(q, (k/temperature).transpose(1, 2))
        attn = self.f(attn)
        out = torch.bmm(attn, v).view(ba, -1)

        if self.use_dueling:  # Dueling DQN
            Q, V = self.Q(out), self.V(out)
            out = Q - Q.mean(dim=1, keepdim=True) + V
        else:
            assert False, 'MAM must use dueling DQN architecture.'
        if show_grid:
            return out, state, attn.detach()
        else:
            return out, state


class SelfAttentionNetNoV(nn.Module):
    def __init__(self, output_shape,
                 em_input_shape, state_input_shape,
                 task_num: int,
                 hidden_type='GCN', graph=None,
                 dueling_param=None, device='cuda',
                 ):

        super().__init__()

        self.em_input_shape = em_input_shape
        self.state_input_shape = state_input_shape
        self.output_shape = output_shape
        self.device = device
        self.g = graph
        self.use_dueling = dueling_param is not None
        self.task_num = task_num
        self.f = nn.Softmax(dim=2)
        self.is_gcn = False

        if hidden_type == 'MLP':
            self.w_k = nn.Sequential(OrderedDict([
                ('k1', nn.Linear(int(state_input_shape), 128, bias=False)),
                ('kRelu', nn.ReLU()),
                ('k2', nn.Linear(128, 64, bias=False))
            ]))

        elif hidden_type == 'GCN':
            self.is_gcn = True
            self.w_k = GCNlayer(in_feats=4, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
        else:
            raise Exception('Invalid layer type, must be either "GCN" or "MLP"')

        self.w_q = nn.Sequential(OrderedDict([
            ('q1', nn.Linear(em_input_shape, 128)),
            ('qRelu', nn.ReLU()),
            ('q2', nn.Linear(128, 64))
        ]))

        if self.use_dueling:  # dueling DQN
            Q_kwargs, V_kwargs = dueling_param

            Q_hidden_size = Q_kwargs['hidden_sizes'][0]
            V_hidden_size = V_kwargs['hidden_sizes'][0]
            self.Q = nn.Sequential(OrderedDict([
                ('Q1', nn.Linear(118, Q_hidden_size)),
                ('QRelu', nn.ReLU()),
                ('Q2', nn.Linear(Q_hidden_size, output_shape))
            ]))
            self.V = nn.Sequential(OrderedDict([
                ('v1', nn.Linear(118, V_hidden_size)),
                ('VRelu', nn.ReLU()),
                ('V2', nn.Linear(V_hidden_size, 1))
            ]))
        else:
            assert False, 'MAM must use dueling DQN architecture.'

    def forward(self, x, state=None, show_grid=False, info=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        state_in, embedding_in = torch.split(x, [self.state_input_shape, self.em_input_shape*self.task_num], dim=1)
        embedding_in = torch.reshape(embedding_in, [-1, self.task_num, self.em_input_shape])

        ba = state_in.shape[0]
        bus_num = int(self.state_input_shape/4)

        q = self.w_q(embedding_in)
        q = torch.mean(q, dim=1).reshape(-1, 1, 64)

        if self.is_gcn:
            bg = dgl.batch([self.g for _ in range(ba)]).to(self.device)
            state_in = state_in.reshape(-1, 4)
            k = self.w_k(bg, state_in).reshape(-1, bus_num, 64)
        else:
            k = self.w_k(state_in).reshape(ba, -1, 64)

        temperature = k.shape[0] ** 0.5

        attn = torch.bmm(q, (k/temperature).transpose(1, 2))
        out = self.f(attn).reshape(ba, -1)

        if self.use_dueling:  # Dueling DQN
            Q, V = self.Q(out), self.V(out)
            out = Q - Q.mean(dim=1, keepdim=True) + V
        else:
            assert False, 'MAM must use dueling DQN architecture.'
        if show_grid:
            return out, state, attn.detach()
        else:
            return out, state


class SelfAttentionNetSingleGCN(nn.Module):
    def __init__(self, output_shape,
                 em_input_shape, state_input_shape,
                 task_num: int,
                 hidden_type='GCN', graph=None,
                 dueling_param=None, device='cuda',
                 ):

        super().__init__()

        self.em_input_shape = em_input_shape
        self.state_input_shape = state_input_shape
        self.output_shape = output_shape
        self.device = device
        self.g = graph
        self.use_dueling = dueling_param is not None
        self.task_num = task_num
        self.f = nn.Softmax(dim=2)
        self.is_gcn = False

        if hidden_type == 'MLP':
            self.w_k_v = nn.Sequential(OrderedDict([
                ('k1', nn.Linear(int(state_input_shape), 128, bias=False)),
                ('kRelu', nn.ReLU()),
                ('k2', nn.Linear(128, 64, bias=False))
            ]))
        elif hidden_type == 'GCN':
            self.is_gcn = True
            self.w_k_v = GCNlayer(in_feats=4, n_hidden=64, n_classes=64, n_layers=2, device=self.device)
        else:
            raise Exception('Invalid layer type, must be either "GCN" or "MLP"')

        self.w_q = nn.Sequential(OrderedDict([
            ('q1', nn.Linear(em_input_shape, 128)),
            ('qRelu', nn.ReLU()),
            ('q2', nn.Linear(128, 64))
        ]))

        if self.use_dueling:  # dueling DQN
            Q_kwargs, V_kwargs = dueling_param
            Q_hidden_size = Q_kwargs['hidden_sizes'][0]
            V_hidden_size = V_kwargs['hidden_sizes'][0]
            self.Q = nn.Sequential(OrderedDict([
                ('Q1', nn.Linear(64, Q_hidden_size)),
                ('QRelu', nn.ReLU()),
                ('Q2', nn.Linear(Q_hidden_size, output_shape))
            ]))
            self.V = nn.Sequential(OrderedDict([
                ('V1', nn.Linear(64, V_hidden_size)),
                ('VRelu', nn.ReLU()),
                ('V2', nn.Linear(V_hidden_size, 1))
            ]))
        else:
            assert False, 'MAM must use dueling DQN architecture.'

    def forward(self, x, state=None, show_grid=False, info=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        state_in, embedding_in = torch.split(x, [self.state_input_shape, self.em_input_shape*self.task_num], dim=1)
        embedding_in = torch.reshape(embedding_in, [-1, self.task_num, self.em_input_shape])

        ba = state_in.shape[0]
        bus_num = int(self.state_input_shape/4)

        q = self.w_q(embedding_in)

        q = torch.mean(q, dim=1).reshape(-1, 1, 64)

        if self.is_gcn:
            bg = dgl.batch([self.g for _ in range(ba)]).to(self.device)
            state_in = state_in.reshape(-1, 4)
            k = self.w_k_v(bg, state_in).reshape(-1, bus_num, 64)
            v = self.w_k_v(bg, state_in).reshape(-1, bus_num, 64)
        else:
            k = self.w_k_v(state_in).reshape(ba, -1, 64)
            v = self.w_k_v(state_in).reshape(ba, -1, 64)

        temperature = k.shape[0] ** 0.5

        attn = torch.bmm(q, (k/temperature).transpose(1, 2))
        attn = self.f(attn)
        out = torch.bmm(attn, v).view(ba, -1)

        if self.use_dueling:  # Dueling DQN
            Q, V = self.Q(out), self.V(out)
            out = Q - Q.mean(dim=1, keepdim=True) + V
        else:
            assert False, 'MAM must use dueling DQN architecture.'
        if show_grid:
            return out, state, attn.detach()
        else:
            return out, state
