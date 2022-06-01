import torch
import dgl
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import hinge_loss



class H2FDetector_layer(nn.Module):
    def __init__(self, input_dim, output_dim, head, relation_aware, etype, dropout, if_sum=False):
        super().__init__()
        self.etype = etype
        self.head = head
        self.hd = output_dim
        self.if_sum = if_sum
        self.relation_aware = relation_aware
        self.w_liner = nn.Linear(input_dim, output_dim*head)
        self.atten = nn.Linear(2*self.hd, 1)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['feat'] = h
            g.apply_edges(self.sign_edges, etype=self.etype)
            h = self.w_liner(h)
            g.ndata['h'] = h
            g.update_all(message_func=self.message, reduce_func=self.reduce, etype=self.etype)
            out = g.ndata['out']
            return out

    def message(self, edges):
        src = edges.src
        src_features = edges.data['sign'].view(-1,1)*src['h']
        src_features = src_features.view(-1, self.head, self.hd)
        z = torch.cat([src_features, edges.dst['h'].view(-1, self.head, self.hd)], dim=-1)
        alpha = self.atten(z)
        alpha = self.leakyrelu(alpha)
        return {'atten':alpha, 'sf':src_features}

    def reduce(self, nodes):
        alpha = nodes.mailbox['atten']
        sf = nodes.mailbox['sf']
        alpha = self.softmax(alpha)
        out = torch.sum(alpha*sf, dim=1)
        if not self.if_sum:
            out = out.view(-1, self.head*self.hd)
        else:
            out = out.sum(dim=-2)
        return {'out':out}

    def sign_edges(self, edges):
        src = edges.src['feat']
        dst = edges.dst['feat']
        score = self.relation_aware(src, dst)
        return {'sign':torch.sign(score)}


class RelationAware(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.d_liner = nn.Linear(input_dim, output_dim)
        self.f_liner = nn.Linear(3*output_dim, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, dst):
        src = self.d_liner(src)
        dst = self.d_liner(dst)
        diff = src-dst
        e_feats = torch.cat([src, dst, diff], dim=1)
        e_feats = self.dropout(e_feats)
        score = self.f_liner(e_feats).squeeze()
        score = self.tanh(score)
        return score


class MultiRelationH2FDetectorLayer(nn.Module):
    def __init__(self, input_dim, output_dim, head, dataset, dropout, if_sum=False):
        super().__init__()
        self.relation = copy.deepcopy(dataset.etypes)
        self.relation.remove('homo')
        self.n_relation = len(self.relation)
        if not if_sum:
            self.liner = nn.Linear(self.n_relation*output_dim*head, output_dim*head)
        else:
            self.liner = nn.Linear(self.n_relation*output_dim, output_dim)
        self.relation_aware = RelationAware(input_dim, output_dim*head, dropout)
        self.minelayers = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)
        for e in self.relation:
            self.minelayers[e] = H2FDetector_layer(input_dim, output_dim, head, self.relation_aware, e, dropout, if_sum)
    
    
    def forward(self, g, h):
        hs = []
        for e in self.relation:
            he = self.minelayers[e](g, h)
            hs.append(he)
        h = torch.cat(hs, dim=1)
        h = self.dropout(h)
        h = self.liner(h)
        return h
    
    def loss(self, g, h):
        with g.local_scope():
            g.ndata['feat'] = h
            agg_h = self.forward(g,h)

            g.apply_edges(self.score_edges, etype='homo')
            edges_score = g.edges['homo'].data['score']
            edge_train_mask = g.edges['homo'].data['train_mask'].bool()
            edge_train_label = g.edges['homo'].data['label'][edge_train_mask]
            edge_train_pos = edge_train_label == 1
            edge_train_neg = edge_train_label == -1
            edge_train_pos_index = edge_train_pos.nonzero().flatten().detach().cpu().numpy()
            edge_train_neg_index = edge_train_neg.nonzero().flatten().detach().cpu().numpy()
            edge_train_pos_index = np.random.choice(edge_train_pos_index, size=len(edge_train_neg_index))
            index = np.concatenate([edge_train_pos_index, edge_train_neg_index])
            index.sort()
            edge_train_score = edges_score[edge_train_mask]
            # hinge loss
            edge_diff_loss = hinge_loss(edge_train_label[index], edge_train_score[index])

            train_mask = g.ndata['train_mask'].bool()
            train_h = agg_h[train_mask]
            train_label = g.ndata['label'][train_mask]
            train_pos = train_label==1
            train_neg = train_label==0
            train_pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
            train_neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
            train_neg_index = np.random.choice(train_neg_index, size=len(train_pos_index))
            node_index = np.concatenate([train_neg_index, train_pos_index])
            node_index.sort()
            pos_prototype = torch.mean(train_h[train_pos], dim=0).view(1,-1)
            neg_prototype = torch.mean(train_h[train_neg], dim=0).view(1,-1)
            train_h_loss = train_h[node_index]
            pos_prototypes = pos_prototype.expand(train_h_loss.shape)
            neg_prototypes = neg_prototype.expand(train_h_loss.shape)
            diff_pos = - F.pairwise_distance(train_h_loss, pos_prototypes)
            diff_neg = - F.pairwise_distance(train_h_loss, neg_prototypes)
            diff_pos = diff_pos.view(-1,1)
            diff_neg = diff_neg.view(-1,1)
            diff = torch.cat([diff_neg, diff_pos], dim=1)
            diff_loss = F.cross_entropy(diff, train_label[node_index])

            return agg_h, edge_diff_loss, diff_loss
            
    
    def score_edges(self, edges):
        src = edges.src['feat']
        dst = edges.dst['feat']
        score = self.relation_aware(src, dst)
        return {'score':score}


class H2FDetector(nn.Module):
    def __init__(self, args, g):
        super().__init__()
        self.n_layer = args.n_layer
        self.input_dim = g.nodes['r'].data['feature'].shape[1]
        self.intra_dim = args.intra_dim
        self.n_class = args.n_class
        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2
        self.n_layer = args.n_layer
        self.mine_layers = nn.ModuleList()
        if args.n_layer == 1:
            self.mine_layers.append(MultiRelationH2FDetectorLayer(self.input_dim, self.n_class, args.head, g, args.dropout, if_sum=True))
        else:
            self.mine_layers.append(MultiRelationH2FDetectorLayer(self.input_dim, self.intra_dim, args.head, g, args.dropout))
            for _ in range(1, self.n_layer-1):
                self.mine_layers.append(MultiRelationH2FDetectorLayer(self.intra_dim*args.head, self.intra_dim, args.head, g, args.dropout))
            self.mine_layers.append(MultiRelationH2FDetectorLayer(self.intra_dim*args.head, self.n_class, args.head, g, args.dropout, if_sum=True))
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, g):
        feats = g.ndata['feature'].float()
        h = self.mine_layers[0](g, feats)
        if self.n_layer > 1:
            h = self.relu(h)
            h = self.dropout(h)
            for i in range(1, len(self.mine_layers)-1):
                h = self.mine_layers[i](g, h)
                h = self.relu(h)
                h = self.dropout(h)
            h = self.mine_layers[-1](g, h)
        return h 
    
    def loss(self, g):
        feats = g.ndata['feature'].float()
        train_mask = g.ndata['train_mask'].bool()
        train_label = g.ndata['label'][train_mask]
        train_pos = train_label == 1
        train_neg = train_label == 0
        
        pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
        neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
        neg_index = np.random.choice(neg_index, size=len(pos_index), replace=False)
        index = np.concatenate([pos_index, neg_index])
        index.sort()
        h, edge_loss, prototype_loss = self.mine_layers[0].loss(g, feats)
        if self.n_layer > 1:
            h = self.relu(h)
            h = self.dropout(h)
            for i in range(1, len(self.mine_layers)-1):
                h, e_loss, p_loss = self.mine_layers[i].loss(g, h)
                h = self.relu(h)
                h = self.dropout(h)
                edge_loss += e_loss
                prototype_loss += p_loss
            h, e_loss, p_loss = self.mine_layers[-1].loss(g, h)
            edge_loss += e_loss
            prototype_loss += p_loss
        model_loss = F.cross_entropy(h[train_mask][index], train_label[index])
        loss = model_loss + self.gamma1*edge_loss + self.gamma2*prototype_loss
        return loss




