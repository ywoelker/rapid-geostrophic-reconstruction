import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.cluster import KMeans
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


class DistanceAssignerModule(nn.Module):
    def __init__(self, n_compartments, centers = None, sigmas_learnable = True):
        super(DistanceAssignerModule, self).__init__()
        self.n_compartments = n_compartments
        
        if centers is not None:
            self.centers = nn.Parameter(torch.tensor(centers), requires_grad = True)
        else:
            self.centers = nn.Parameter((torch.rand(self.n_compartments, 3) * torch.tensor([1,1,2]) - torch.tensor([0,0,1])), requires_grad = True)
        
        # self.sigma = nn.Parameter(torch.ones(1, self.n_compartments) * .1, requires_grad=True)
        self.sigmas = nn.Parameter(torch.ones(1, self.n_compartments, 3) * 1, requires_grad=sigmas_learnable)



    def forward(self, x):

        if len(x.shape) == 3:

            centers = self.centers.unsqueeze(0).unsqueeze(0)
            sigmas = torch.abs(self.sigmas.unsqueeze(0)) + 1e-6
        else:
            centers = self.centers.unsqueeze(0)
            sigmas = torch.abs(self.sigmas) + 1e-6

        dist = ((x.unsqueeze(-2) - centers) ** 2 / sigmas ** 2).sum(dim = -1)

        return torch.softmax(- dist, dim = -1)
        # return torch.softmax(- torch.cdist(x, self.centers) ** 2 / self.sigma ** 2, dim = -1)
        
        
   
    


    def get_adjacent_matrix(self):
        return torch.softmax(-torch.cdist(self.centers, self.centers) ** 2, dim = -1)
    

class AssignerModule(nn.Module):

    def __init__(self, lon_inner_boundaries, day_inner_boundaries):
        super(AssignerModule, self).__init__()

        self.lon_inner_boundaries = lon_inner_boundaries
        self.day_inner_boundaries = day_inner_boundaries

        assert len(lon_inner_boundaries) > 0 and len(day_inner_boundaries) > 0 and len(lon_inner_boundaries) == len(day_inner_boundaries)

        self.num_buckets = len(lon_inner_boundaries) - 1

        print(self.num_buckets)
        
    def forward(self, x):
        
        lon = x[..., 0]
        lat = x[..., 1]
        day = x[..., 2]
    
        lon_buckets = torch.bucketize(lon, self.lon_inner_boundaries)
        day_buckets = torch.bucketize(day, self.day_inner_boundaries)

        def merge_boundary_buckets(values):
            values = torch.minimum(values, torch.tensor( self.num_buckets))
            values = torch.maximum(values, torch.tensor(1)) -1 
            return values
        
        lon_buckets = merge_boundary_buckets(lon_buckets)
        day_buckets = merge_boundary_buckets(day_buckets)

        buckets = lon_buckets * self.num_buckets + day_buckets

        assignment = torch.nn.functional.one_hot(buckets, num_classes = self.num_buckets ** 2).float()
        return assignment.detach()
    


class BysGNN_GNN_GCN(nn.Module):
    def __init__(self, node_feature_dim,
                 device='cpu', conv_hidden_dim=64, conv_num_layers=3, gcn_type = 'GCN'):
        super(BysGNN_GNN_GCN, self).__init__()
        self.node_feature_dim = node_feature_dim      
        self.conv_hidden_dim = conv_hidden_dim
        self.conv_layers_num = conv_num_layers

        self.convs = nn.ModuleList()
        if gcn_type == 'GCN':
            self.convs.append( pyg_nn.GCNConv(self.node_feature_dim, self.conv_hidden_dim))
        elif gcn_type == 'Cheb':
            self.convs.append(
                # pyg_nn.GCNConv(self.node_feature_dim, self.conv_hidden_dim, add_self_loops=False, normalize=False)
                # pyg_nn.dense.DenseGCNConv(self.node_feature_dim, self.conv_hidden_dim)
                pyg_nn.ChebConv(self.node_feature_dim, self.conv_hidden_dim, K=3, normalization='rw')
            ) 
        self.layer_norms = nn.ModuleList(
            # [nn.Identity()] + 
            [nn.LayerNorm(self.conv_hidden_dim) for _ in range(self.conv_layers_num-1)]
        )
        for l in range(self.conv_layers_num-1):
            if gcn_type == 'GCN':
                self.convs.append( pyg_nn.GCNConv(self.conv_hidden_dim, self.conv_hidden_dim))
            elif gcn_type == 'Cheb':
                self.convs.append(
                    # pyg_nn.dense.DenseGCNConv(self.conv_hidden_dim, self.conv_hidden_dim)
                    pyg_nn.ChebConv(self.conv_hidden_dim, self.conv_hidden_dim, K=3, normalization='rw')
                )
        
        # # initialize weights using xavier
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.GRU):
        #         for name, param in m.named_parameters():
        #             if 'weight' in name:
        #                 nn.init.xavier_normal_(param)
        #             elif 'bias' in name:
        #                 param.data.zero_()
        #     elif isinstance(m, nn.LayerNorm):
        #         m.bias.data.zero_()
        #         m.weight.data.fill_(1.0)
        # self.to(device)


    # def forward(self, X, adj_mat):
    #     """
    #     We have a different adjacency matrix for each batch element.
    #     This is not covered in the torch_geometric library. (at least if we want to use more than the default GCNConv (torch_geometric.nn.dense.DenseGCNConv))
    #     Therefore, we need to iterate over the batch and apply the GCNConv for each batch element separately.
    #     """

    #     n_batch = X.shape[0]
    #     n_sensors = X.shape[1]

    #     if len(adj_mat.shape) == 2:
    #         adj_mat = adj_mat.unsqueeze(0)

    #     assert len(adj_mat.shape) == 3, 'adj_mat should have shape (batch, n_sensors, n_sensors) or (n_sensors, n_sensors)'
        
    #     if adj_mat.shape[0] == 1 and n_batch > 1:
    #         adj_mat = adj_mat.repeat(n_batch, 1, 1)

    #     X_gnn = torch.zeros((n_batch, n_sensors, self.conv_hidden_dim)).to(X.device)
    #     for i in range(n_batch):
    #         edge_indices, edge_attrs = pyg_utils.dense_to_sparse(adj_mat[i])
    #         X_gnn_i = self.convs[0](X[i], edge_indices, edge_attrs)
    #         X_gnn_i = F.relu(X_gnn_i)
    #         for layers_i in range(1, self.conv_layers_num):
    #             X_res = X_gnn_i # Store the current state for the residual connection
    #             X_gnn_i = self.convs[layers_i](X_gnn_i, edge_indices, edge_attrs)
    #             X_gnn_i = F.relu(self.layer_norms[layers_i](X_gnn_i + X_res))
    #         X_gnn[i] = X_gnn_i
    #     return X_gnn


    def forward(self, X, adj_mat):
        assert len(adj_mat.shape) == 2, 'adj_mat should have shape (n_sensors, n_sensors)'
        edge_indices, edge_attrs = pyg_utils.dense_to_sparse(adj_mat)
        X_gnn = self.convs[0](X, edge_indices, edge_attrs)
        X_gnn = F.relu(X_gnn)
        for stack_i in range(1, self.conv_layers_num):
            X_res = X_gnn # Store the current state for the residual connection
            X_gnn = self.convs[stack_i](X_gnn, edge_indices, edge_attrs)
            X_gnn = F.relu(self.layer_norms[stack_i-1](X_gnn + X_res)) # Add the residual connection

        return X_gnn



class ProfileModelSUSTeR5_fast(nn.Module):
    def __init__(self, n_features, n_compartments, n_embedding, n_dvdz_features, z_scale, device,
                profile_embedder = None, node_assigner = None, 
                argo_mean_in_embedding = False, argo_mean_in_gnn_input = True,
                assignment_threshold = 0.05, embedding_dropout = .22, adj_threshold = 0.1,  num_gcn_layers = 3, gcn_type = 'GCN'):
        super(ProfileModelSUSTeR5_fast, self).__init__()
        self.n_compartments = n_compartments #* n_compartments
        self.n_features = n_features
        self.n_dvdz_features = n_dvdz_features
        self.n_embedding = n_embedding
        self.argo_mean_in_embedding = argo_mean_in_embedding
        self.argo_mean_in_gnn_input = argo_mean_in_gnn_input
        self.assignment_threshold = assignment_threshold
        self.adj_threshold = adj_threshold
        # self.fc1 = nn.Linear(n_features+n_embedding * self.n_compartments +n_dvdz_features +4 +3 + 12, 64)
        self.fc1 = nn.Linear(n_features + n_embedding * self.n_compartments + 12 + n_dvdz_features, 64)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(64,64)
        self.dropout1 = nn.Dropout(p=.25)
        self.fc3 = nn.Linear(64,64)
        self.dropout2 = nn.Dropout(p=.25)
        self.fc4 = nn.Linear(64,64)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.dropout3 = nn.Dropout(p=0)
        self.fc5 = nn.Linear(64,32)
        self.fc6 = nn.Linear(32,1)

        self.profile_embedder = profile_embedder

        self.compartment_models = nn.ModuleList([
            nn.Sequential(
                # nn.Linear(n_features , n_embedding * 2), 
                nn.Linear(n_features +3, n_embedding * 2), 
                nn.ReLU(),
                # nn.Dropout(p=.25),    
                nn.Linear(n_embedding * 2, n_embedding), 
                # nn.LayerNorm(n_embedding),
                nn.Tanh(), 

            ) for _ in range(self.n_compartments)])
        
        self.compartment_gate = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features + 3, n_embedding ),
                nn.ReLU(),
                nn.Linear(n_embedding, n_embedding),
                nn.Sigmoid(),
            ) for _ in range(self.n_compartments)])

        self.E1 = nn.Parameter(torch.randn(n_compartments, 8), requires_grad = True)
        self.E2 = nn.Parameter(torch.randn(n_compartments, 8), requires_grad = True)

        # self.transformer = PredictionToTransportModel(z_scale, torch.tensor(dv_dz_obs.dv_dz_times_X.mean('time').fillna(0).values).to(device), torch.tensor(dv_dz_obs.dv_dz_times_X.std('time').fillna(1).values).to(device))

        self.gcn = BysGNN_GNN_GCN(n_embedding, device, n_embedding, num_gcn_layers, gcn_type= gcn_type)

        if node_assigner is not None:
            self.node_assigner = node_assigner
        else:
            self.node_assigner = DistanceAssignerModule(n_compartments)
        self.device = device

        self.alpha = nn.Parameter(torch.tensor(.5), requires_grad = False)

        self.x_gnn_norm = nn.LayerNorm(n_embedding)

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)   

        self.key_linear = nn.Linear(n_embedding, n_embedding)
        self.query_linear = nn.Linear(n_embedding, n_embedding)

        self.mean_cluster_init_distance = torch.mean(self.node_assigner.get_adjacent_matrix().detach())
        # print('mean_cluster_init_distance', self.mean_cluster_init_distance)

    def forward(self, x, mask_padded, lon, lat, dv_dz, days, y_prev, missing_indices, fs, ac, ws):

        # x shape (batch, 4, n_profiles, n_features) -- n_profiles variable
        # lon shape (batch, 4, n_profiles) -- n_profiles variable
        # lat shape (batch, 4, n_profiles) -- n_profiles variable
        # dv_dz shape (batch, 142)
        # days shape (batch, 4, n_profiles) -- n_profiles variable
        # y_prev shape (batch, 3)
        # missing_indices shape (batch, 4)


        
        sum_mask_padded = torch.sum(mask_padded, dim=1, keepdim=True)
        sum_input = torch.sum(x, dim=1)
        argo_mean = torch.where(sum_mask_padded > 0, sum_input / sum_mask_padded, torch.zeros_like(sum_input))


        if not self.argo_mean_in_gnn_input:
            x = x - argo_mean.unsqueeze(1)

        assignment_input = torch.stack([lon, lat, days], dim = -1) # shape (batch, max_profiles, 3)



        node_assignments = self.node_assigner(assignment_input) # shape (batch, max_profiles, n_compartments)

        assert not torch.isnan(node_assignments).any(), 'NaN in fresh node_assignment'

        node_assignments = node_assignments * mask_padded.unsqueeze(-1) # shape (batch, max_profiles, n_compartments)

        mask = node_assignments >= self.assignment_threshold # shape (batch, max_profiles, n_compartments)

        node_assignments = node_assignments * mask # shape (batch, max_profiles, n_compartments)

        assert not torch.isnan(node_assignments).any(), 'NaN in mask'


        input_to_layer = torch.cat(
            (x, 
             lon.unsqueeze(-1),
             lat.unsqueeze(-1),
             days.unsqueeze(-1),
             ), dim = -1) # shape (batch, max_profiles, n_features + 3)
        
        weighted_mean = torch.sum(input_to_layer.unsqueeze(-2) * node_assignments.unsqueeze(-1), axis = 1) / (torch.sum(node_assignments, axis = 1).unsqueeze(-1) + 1e-6) # shape (batch, n_compartments, n_features + 3)

        weighted_mean = torch.cat(
            (
                weighted_mean, 
                # missing_indices.unsqueeze(1).repeat(1, self.n_compartments, 1),
            ), dim = -1
        ) # shape (batch, n_compartments, n_features + 3 + 4)

        assert not torch.isnan(weighted_mean).any(), 'NaN in weighted_mean'
        assert self.n_compartments == weighted_mean.shape[1], 'n_compartments != weighted_mean.shape[1]'
        compartment_embeddings = []
        for i in range(self.n_compartments):
            compartment_embeddings.append(
                self.compartment_models[i](weighted_mean[:, i]).unsqueeze(1) * self.compartment_gate[i](weighted_mean[:, i]).unsqueeze(1)
            )

        graphs = torch.cat(compartment_embeddings, dim = 1) # shape (batch, n_compartments, n_embedding)


        # # similarity = torch.bmm(self.key_linear(graphs), self.query_linear(graphs).permute(0, 2, 1)) #- torch.eye(self.n_compartments).to(self.device).unsqueeze(0) # [b, p, p]
        # similarity = torch.bmm(self.E1.unsqueeze(0), self.E2.unsqueeze(0).permute(0,2,1)) #- torch.eye(self.n_compartments).to(self.device).unsqueeze(0) # [b, p, p]

        # # Compute the magnitudes of each embedding vector
        # magnitude = torch.norm(graphs, p= 2, dim= -1, keepdim=True) + 1e-10

        # # Normalize the dot product by the magnitudes
        # normalized_similarity_matrix = similarity / (magnitude * magnitude.transpose(1, 2))

        # # if a row is all zeros, then the cosine similarity will be NaN, so we replace it with zeros
        # cosine_similarities = normalized_similarity_matrix.mean(dim=0).abs()
        # cosine_similarities[torch.isnan(cosine_similarities)] = 0

        
        
        # def case_amplf_mask(attention, p=2.5, threshold=0.08):
        #     '''
        #     This function computes the case amplification mask for a 2D attention tensor 
        #     with the given amplification factor p.

        #     Parameters:
        #         - attention (torch.Tensor): A 2D attention tensor of shape [n, n].
        #         - p (float): The case amplification factor (default: 2.5).
        #         - threshold (float): The threshold for the mask (default: 0.05).

        #     Returns:
        #         - mask (torch.Tensor): A 2D binary mask of the same size as `attention`,
        #         where 0s denote noisy elements and 1s denote clean elements.
        #     '''
        #     # Compute the maximum value in the attention tensor
        #     max_val, _ = torch.max(attention.detach(), dim=1, keepdim=True)

        #     # Compute the mask as per the case amplification formula
        #     mask = (attention.detach() / max_val) ** p

        #     # Turn the mask into a binary matrix, where anything below threshold will be considered as zero
        #     mask = torch.where(mask > threshold, torch.tensor(1).to(attention.device), torch.tensor(0).to(attention.device))
        #     return mask
        
        # mask = case_amplf_mask(cosine_similarities, p=2.5, threshold=0.18)

        # cosine_similarities = cosine_similarities * mask

        similarity = torch.mm(self.E1, self.E2.permute(1,0)) #- torch.eye(self.n_compartments).to(self.device).unsqueeze(0) # [b, p, p]
        similarity = torch.relu(similarity)
        learned_matrix = torch.softmax(similarity, dim = -1)
        learned_matrix = learned_matrix#.unsqueeze(0)

        cosine_similarities = self.node_assigner.get_adjacent_matrix().detach()
        cosine_similarities = torch.exp(-cosine_similarities**2 / (2 * (self.mean_cluster_init_distance**2)))
        cosine_similarities = cosine_similarities#.unsqueeze(0)

        adjacency_matrix = self.alpha * cosine_similarities + (1- self.alpha) * learned_matrix

        adjacency_matrix[adjacency_matrix < self.adj_threshold] = 0


        assert not torch.isnan(graphs).any(), 'NaN in graphs'

        X_raw = graphs
        X_gnn = self.gcn(X_raw, adjacency_matrix) # [b, p, e]

        # X_raw = self.raw_map(X_raw) 
        # X_hat = self.x_gnn_norm(X_gnn + X_raw)
        # X_hat = self.x_gnn_norm(X_gnn)
        # X_hat = self.x_gnn_norm(X_gnn)
        X_hat = self.embedding_dropout(X_gnn)   

        assert not torch.isnan(X_hat.reshape(X_gnn.shape[0], -1)).any(), 'NaN in X_hat'

        ## Only sort scales (florida current, and windstress)
        # embedding_all_profiles = torch.concat(
        #     [torch.zeros_like(X_hat.reshape(X_gnn.shape[0], -1))] +
        #     [torch.zeros_like(torch.stack(dv_dz, axis = 0))] + 
        #     [torch.zeros_like(missing_indices)] +
        #     [torch.zeros_like(y_prev)] + 
        #     [torch.zeros_like(ac.reshape(-1,1))]+
        #     [fs.reshape(-1,1), ws] 
        # ,dim = -1) # [b, p*e+delta]

        ## Only geostorphic scales (argo, mooring, and antilles current)
        # embedding_all_profiles = torch.concat(
        #     [X_hat.reshape(X_gnn.shape[0], -1)] +
        #     [torch.stack(dv_dz, axis = 0)] + 
        #     [missing_indices] +
        #     [y_prev] + 
        #     [ac.reshape(-1,1)] +
        #     [
        #         torch.zeros_like(fs.reshape(-1,1)),
        #         torch.zeros_like(ws)
        #     ] 
        # ,dim = -1) # [b, p*e+delta]

        # All scales 
        embedding_all_profiles = torch.concat(
            [argo_mean if self.argo_mean_in_embedding else torch.zeros_like(argo_mean)]+
            [X_hat.reshape(X_gnn.shape[0], -1)] +
            [torch.stack(dv_dz, axis = 0)] + 
            # [missing_indices] +
            # [y_prev] + 
            [
                fs.reshape(-1,1),
                ac.reshape(-1,1), 
                ws
            ] 
        ,dim = -1) # [b, p*e+delta]

        
        assert not torch.isnan(embedding_all_profiles).any(), 'NaN in embedding_all_profiles'
        # embedding_all_profiles = self.dropout1(embedding_all_profiles)
        x = torch.relu(self.fc1(embedding_all_profiles))
        # x = self.bn1(x)
        # x = torch.relu(self.fc2(x))
        # x = self.dropout2(x)
        # x = torch.relu(self.fc3(x))
        # x = self.dropout3(x)
        # x = torch.relu(self.fc4(x))
        # x = self.bn2(x)
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        x = x.squeeze(dim = -1)

        return x,( X_hat, embedding_all_profiles)
    

class ProfileModelSUSTeR6_wind(nn.Module):

    def __init__(self, n_features, n_compartments, n_embedding, n_dvdz_features, z_scale, device, profile_embedder = None, node_assigner = None, use_argo_mean = True, use_argo_std = False, use_auxiliry = True) -> None:
        super(ProfileModelSUSTeR6_wind, self).__init__()
        
        self.use_argo_mean = use_argo_mean
        self.use_argo_std = use_argo_std
        self.use_auxiliry = use_auxiliry
        input_size = (n_features if use_argo_mean else 0) + (n_features if use_argo_std else 0) + (10 if use_auxiliry else 0)

        assert input_size > 0


        self.l1 = nn.Linear(input_size, 64)

        self.l2 = nn.Linear(64,32)

        self.l3 = nn.Linear(32,1)
        
        

    def forward(self, x, mask_padded, lon, lat, dv_dz, days, y_prev, missing_indices, fs, ac, ws):

        # x shape (batch, 4, n_profiles, n_features) -- n_profiles variable
        # lon shape (batch, 4, n_profiles) -- n_profiles variable
        # lat shape (batch, 4, n_profiles) -- n_profiles variable
        # dv_dz shape (batch, 142)
        # days shape (batch, 4, n_profiles) -- n_profiles variable
        # y_prev shape (batch, 3)
        # missing_indices shape (batch, 4)


        

        argo_mean = torch.sum(x, dim = 1) / torch.sum(mask_padded, dim = 1, keepdim = True)
        argo_std = torch.sqrt(((x - argo_mean.unsqueeze(1)) ** 2).sum(dim = 1) / (torch.sum(mask_padded, dim=1, keepdim=True) - 1))

        if self.use_auxiliry:
            input_list = [               
                # fs.unsqueeze(-1),
                # ac.unsqueeze(-1),
                ws,
            ]
        else:
            input_list = []
    
        if self.use_argo_mean:
            input_list.append(argo_mean)

        if self.use_argo_std:
            input_list.append(argo_std)

        input = torch.cat( input_list,dim = -1)

        x = self.l1(input)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)

        x = self.l3(x)
        x = x.squeeze(dim = -1)

        return x, None

class ProfileModelSUSTeR6(nn.Module):

    def __init__(self, n_features, n_compartments, n_embedding, n_dvdz_features, z_scale, device, profile_embedder = None, node_assigner = None, use_argo_mean = True, use_argo_std = False, use_auxiliry = True) -> None:
        super(ProfileModelSUSTeR6, self).__init__()
        
        self.use_argo_mean = use_argo_mean
        self.use_argo_std = use_argo_std
        self.use_auxiliry = use_auxiliry
        input_size = (n_features if use_argo_mean else 0) + (n_features if use_argo_std else 0) + (12 if use_auxiliry else 0)

        assert input_size > 0


        self.l1 = nn.Linear(input_size, 64)

        self.l2 = nn.Linear(64,32)

        self.l3 = nn.Linear(32,1)
        
        

    def forward(self, x, mask_padded, lon, lat, dv_dz, days, y_prev, missing_indices, fs, ac, ws):

        # x shape (batch, 4, n_profiles, n_features) -- n_profiles variable
        # lon shape (batch, 4, n_profiles) -- n_profiles variable
        # lat shape (batch, 4, n_profiles) -- n_profiles variable
        # dv_dz shape (batch, 142)
        # days shape (batch, 4, n_profiles) -- n_profiles variable
        # y_prev shape (batch, 3)
        # missing_indices shape (batch, 4)


        

        argo_mean = torch.sum(x, dim = 1) / torch.sum(mask_padded, dim = 1, keepdim = True)
        argo_std = torch.sqrt(((x - argo_mean.unsqueeze(1)) ** 2).sum(dim = 1) / (torch.sum(mask_padded, dim=1, keepdim=True) - 1))

        if self.use_auxiliry:
            input_list = [               
                fs.unsqueeze(-1),
                ac.unsqueeze(-1),
                ws,
            ]
        else:
            input_list = []
    
        if self.use_argo_mean:
            input_list.append(argo_mean)

        if self.use_argo_std:
            input_list.append(argo_std)

        input = torch.cat( input_list,dim = -1)

        x = self.l1(input)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)

        x = self.l3(x)
        x = x.squeeze(dim = -1)

        return x, None
    
class ProfileModelSUSTeR7(nn.Module):
    def __init__(self, n_features, n_compartments, n_embedding, n_dvdz_features, z_scale, device,
                profile_embedder = None, node_assigner = None, 
                argo_mean_in_embedding = False, argo_mean_in_gnn_input = True):
        super(ProfileModelSUSTeR7, self).__init__()
        self.n_compartments = n_compartments #* n_compartments
        self.n_features = n_features
        self.n_dvdz_features = n_dvdz_features
        self.n_embedding = n_embedding
        self.argo_mean_in_embedding = argo_mean_in_embedding
        self.argo_mean_in_gnn_input = argo_mean_in_gnn_input
        # self.fc1 = nn.Linear(n_features+n_embedding * self.n_compartments +n_dvdz_features +4 +3 + 12, 64)
        self.fc1 = nn.Linear(n_features + n_embedding * 3 + 12, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc5 = nn.Linear(64,32)
        self.fc6 = nn.Linear(32,1)

        self.profile_embedder = profile_embedder


        self.E1 = nn.Parameter(torch.randn(n_compartments, 8), requires_grad = True)
        self.E2 = nn.Parameter(torch.randn(n_compartments, 8), requires_grad = True)


        self.gcn = BysGNN_GNN_GCN(n_features+3, device, n_embedding, 3)

        self.device = device

        self.embedding_dropout = nn.Dropout(p=.15796541)   

        self.key_linear = nn.Linear(n_features + 3, n_embedding)
        self.query_linear = nn.Linear(n_features + 3, n_embedding)

    def forward(self, x, mask_padded, lon, lat, dv_dz, days, y_prev, missing_indices, fs, ac, ws):

        # x shape (batch, 4, n_profiles, n_features) -- n_profiles variable
        # lon shape (batch, 4, n_profiles) -- n_profiles variable
        # lat shape (batch, 4, n_profiles) -- n_profiles variable
        # dv_dz shape (batch, 142)
        # days shape (batch, 4, n_profiles) -- n_profiles variable
        # y_prev shape (batch, 3)
        # missing_indices shape (batch, 4)


        
        argo_mean = torch.sum(x, dim = 1) / torch.sum(mask_padded, dim = 1, keepdim = True)

        if not self.argo_mean_in_gnn_input:
            x = x - argo_mean.unsqueeze(1)


        graphs = torch.cat(
            (x, 
             lon.unsqueeze(-1),
             lat.unsqueeze(-1),
             days.unsqueeze(-1),
             ), dim = -1) # shape (batch, max_profiles, n_features + 3)

        output_nodes = torch.zeros_like(graphs)[:, :3,:]

        output_nodes[:, :, -3] = torch.tensor([.1, .5, .9]) # longitudes
        output_nodes[:, :, -2] = torch.tensor([.3, .5, .7]) # latitudes
        output_nodes[:, :, -1] = torch.tensor([.5, .5, .5]) # days

        graphs = torch.cat((graphs, output_nodes), dim = 1)

        adjacency_matrix = torch.bmm(self.key_linear(graphs), self.query_linear(graphs).permute(0, 2, 1)) #- torch.eye(self.n_compartments).to(self.device).unsqueeze(0) # [b, p, p]
        adjacency_matrix = torch.relu(adjacency_matrix)
        cosine_similarities = torch.softmax(adjacency_matrix, dim = -1)


        # # compute the distance between each argo from the coordinates formed from the lon and lat

        # coordinates = torch.stack([lon, lat], dim = -1) # shape (batch, max_profiles, 2)
        # coordinates = torch.cat([coordinates, output_nodes[:, :, -3:-1]], dim = 1) # shape (batch, max_profiles + 3, 2)

        # distance_matrix = torch.cdist(coordinates, coordinates) # shape (batch, max_profiles + 3, max_profiles)
        # # apply a gaussian kernel to the distance matrix with a sigma of 2 longitude degrees
        # adjacency_matrix = torch.exp(- distance_matrix ** 2 / 2 ) # shape (batch, max_profiles, max_profiles)


        # cosine_similarities = adjacency_matrix  
        # cosine_similarities[cosine_similarities < 0.1] = 0


        assert not torch.isnan(graphs).any(), 'NaN in graphs'

        X_raw = graphs
        X_gnn = self.gcn(X_raw, cosine_similarities) # [b, p, e]

        processed_nodes = X_gnn[:, -3:, :]
        processed_nodes  = self.embedding_dropout(processed_nodes)


        # All scales 
        embedding_all_profiles = torch.concat(
            [argo_mean if self.argo_mean_in_embedding else torch.zeros_like(argo_mean)]+
            [processed_nodes.reshape(X_gnn.shape[0], -1)] +
            # [torch.stack(dv_dz, axis = 0)] + 
            # [missing_indices] +
            # [y_prev] + 
            [
                fs.reshape(-1,1),
                ac.reshape(-1,1), 
                ws
            ] 
        ,dim = -1) # [b, p*e+delta]

        
        assert not torch.isnan(embedding_all_profiles).any(), 'NaN in embedding_all_profiles'
        # embedding_all_profiles = self.dropout1(embedding_all_profiles)
        x = torch.relu(self.fc1(embedding_all_profiles))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        x = x.squeeze(dim = -1)

        return x,( X_gnn, embedding_all_profiles)


def init_assignment_module(distance_assignment_module, train_dataset, n_compartments, device):
    if distance_assignment_module == 'distance':


        lons_in_training =       train_dataset.lon_scaled.flatten()
        lats_in_training =       train_dataset.lat_scaled.flatten()
        days_in_training =       train_dataset.days_scaled.flatten()

        mask = ~np.isnan(lons_in_training) & ~np.isnan(lats_in_training) & ~np.isnan(days_in_training)
        lons_in_training_nonnan = lons_in_training[mask]
        lats_in_training_nonnan = lats_in_training[mask]
        days_in_training_nonnan = days_in_training[mask]

        X_lonlatdays = np.stack([lons_in_training_nonnan, lats_in_training_nonnan, days_in_training_nonnan], axis = -1)
        kmeans = KMeans(n_clusters=n_compartments, n_init='auto').fit(X_lonlatdays)
            
        node_assigner = DistanceAssignerModule(n_compartments, kmeans.cluster_centers_, sigmas_learnable= True)
    elif distance_assignment_module == 'fixed' or True:
        node_assigner = AssignerModule(
                lon_inner_boundaries = torch.linspace(0, 1, n_compartments+1).to(device),
                day_inner_boundaries = torch.linspace(0, 1, n_compartments+1).to(device)
            )
        
    return node_assigner