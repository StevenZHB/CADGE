import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.nn.functional as fn
from functools import partial


class GATlayer(nn.Module):
    def __init__(self, in_feat, out_feat, q_size, feat_drop, negative_slope, initializer_range = 0.02):
        super(GATlayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.q_size = q_size
        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.initializer_range = initializer_range
        self.bce = nn.BCEWithLogitsLoss(reduction='none')


        # weight bases in equation (3)
        self.atten1 = nn.Linear(self.in_feat+self.q_size, 1, bias = False)
        self.atten2 = nn.Linear(self.in_feat+self.q_size, 1, bias = False)
        self._init_weights()

    def _init_weights(self):
        """参数初始化"""
        for module in self.modules():
            if isinstance(module, (nn.Linear)):
                module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, graph, catcu_lss=True):
        def message_func(edges):
            return {'msg': edges.src['h'], 'a': edges.data['a'], 't': edges.src['tar'], 'l': edges.src['loss'],'t_num': edges.src['tar_num']}
            # return {'msg': edges.src['h'], 'a': edges.data['a']}

        def attention_message_func_node(edges):
            # print(self.a_l.device)
            h = torch.cat([edges.src['h'], edges.src['q']], dim=1)
            a = self.atten1(h)
            return {'a': self.leaky_relu(a)}

        def apply_func(nodes):
            # print(nodes.mailbox['a'])
            alpha = F.softmax(nodes.mailbox['a'], dim=1)

            if catcu_lss:
                loss = torch.sum(self.bce(nodes.mailbox['a'].squeeze(-1),nodes.mailbox['t']),dim=-1)*(torch.sum(nodes.mailbox['t'],dim=-1)>0)+ torch.sum(nodes.mailbox['l'],dim=-1)
                tar_num = nodes.mailbox['t'].shape[-1]*(torch.sum(nodes.mailbox['t'],dim=-1)>0) + torch.sum(nodes.mailbox['t_num'],dim=-1)

            h = torch.sum(alpha*nodes.mailbox['msg'], dim=1)
            # h = torch.mean(nodes.mailbox['msg'], dim=1)
            return {'h': h, 'loss': loss, 'tar_num': tar_num}

        def attention_message_func_root(edges):
            # print(self.a_l.device)
            h = torch.cat([edges.src['h'], edges.src['q']], dim=1)
            a = self.atten2(h)
            return {'a': self.leaky_relu(a)}

        graph.apply_edges(attention_message_func_node)
        graph.update_all(message_func, apply_func)
        graph.apply_edges(attention_message_func_root)
        graph.update_all(message_func, apply_func)
        '''if catcu_lss:
            loss = torch.sum(graph.ndata['loss'][0])/(graph.num_nodes()-1)
        root_hidden = graph.ndata['h'][0].view((4,-1))
        return root_hidden,loss'''
        return graph