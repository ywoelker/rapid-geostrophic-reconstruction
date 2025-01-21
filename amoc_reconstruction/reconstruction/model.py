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
                 device='cpu', conv_hidden_dim=64, conv_num_layers=3):
        super(BysGNN_GNN_GCN, self).__init__()
        self.node_feature_dim = node_feature_dim      
        self.conv_hidden_dim = conv_hidden_dim
        self.conv_layers_num = conv_num_layers

        self.convs = nn.ModuleList()
        self.convs.append(pyg_nn.GCNConv(self.node_feature_dim, self.conv_hidden_dim, add_self_loops=False, normalize=False)) 
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(self.conv_hidden_dim) for _ in range(self.conv_layers_num-1)
        )
        for l in range(self.conv_layers_num-1):
            self.convs.append(pyg_nn.GCNConv(self.conv_hidden_dim, self.conv_hidden_dim, add_self_loops=False, normalize=False))
        
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

    def forward(self, X, adj_mat):
        edge_indices, edge_attrs = pyg_utils.dense_to_sparse(adj_mat)
        X_gnn = self.convs[0](X, edge_indices, edge_attrs)
        X_gnn = F.relu(X_gnn)
        for stack_i in range(1, self.conv_layers_num):
            X_res = X_gnn # Store the current state for the residual connection
            X_gnn = self.convs[stack_i](X_gnn, edge_indices, edge_attrs)

            X_gnn = F.relu(self.layer_norms[stack_i-1](X_gnn + X_res)) # Add the residual connection
            # X_gnn = F.relu(X_gnn + X_res) # Add the residual connection
        return X_gnn



class ProfileModelSUSTeR5_fast(nn.Module):
    def __init__(self, n_features, n_compartments, n_embedding, n_dvdz_features, z_scale, device,
                profile_embedder = None, node_assigner = None, 
                argo_mean_in_embedding = False, argo_mean_in_gnn_input = True):
        super(ProfileModelSUSTeR5_fast, self).__init__()
        self.n_compartments = n_compartments #* n_compartments
        self.n_features = n_features
        self.n_dvdz_features = n_dvdz_features
        self.n_embedding = n_embedding
        self.argo_mean_in_embedding = argo_mean_in_embedding
        self.argo_mean_in_gnn_input = argo_mean_in_gnn_input
        # self.fc1 = nn.Linear(n_features+n_embedding * self.n_compartments +n_dvdz_features +4 +3 + 12, 64)
        self.fc1 = nn.Linear(n_features + n_embedding * self.n_compartments + 12, 64)
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
                nn.Linear(n_features , n_embedding * 2), 
                # nn.Linear(n_features +3 +4, n_embedding), 
                nn.ReLU(),
                # nn.Dropout(p=.25),    
                nn.Linear(n_embedding * 2, n_embedding), 
                nn.LayerNorm(n_embedding),
                nn.Tanh(), 

            ) for _ in range(self.n_compartments)])

        self.E1 = nn.Parameter(torch.randn(n_compartments, 8), requires_grad = True)
        self.E2 = nn.Parameter(torch.randn(n_compartments, 8), requires_grad = True)

        # self.transformer = PredictionToTransportModel(z_scale, torch.tensor(dv_dz_obs.dv_dz_times_X.mean('time').fillna(0).values).to(device), torch.tensor(dv_dz_obs.dv_dz_times_X.std('time').fillna(1).values).to(device))

        self.gcn = BysGNN_GNN_GCN(n_embedding, device, n_embedding, 1)

        if node_assigner is not None:
            self.node_assigner = node_assigner
        else:
            self.node_assigner = DistanceAssignerModule(n_compartments)
        self.device = device

        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad = False)

        self.self_attn = nn.MultiheadAttention(n_embedding, 1)

        self.raw_map = nn.Linear(n_embedding, n_embedding)
        self.x_gnn_norm = nn.LayerNorm(n_embedding)

        self.embedding_dropout = nn.Dropout(p=.25796541)   

        self.key_linear = nn.Linear(n_embedding, n_embedding)
        self.query_linear = nn.Linear(n_embedding, n_embedding)


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

        assignment_input = torch.stack([lon, lat, days], dim = -1) # shape (batch, max_profiles, 3)


        assert not torch.isnan(assignment_input).any(), 'NaN in assignment_input'

        node_assignments = self.node_assigner(assignment_input) # shape (batch, max_profiles, n_compartments)

        assert not torch.isnan(node_assignments).any(), 'NaN in fresh node_assignment'

        node_assignments = node_assignments * mask_padded.unsqueeze(-1) # shape (batch, max_profiles, n_compartments)

        mask = node_assignments >= .05 # shape (batch, max_profiles, n_compartments)

        node_assignments = node_assignments * mask # shape (batch, max_profiles, n_compartments)

        assert not torch.isnan(node_assignments).any(), 'NaN in mask'
        input_to_layer = torch.cat(
            (x, 
            #  lon.unsqueeze(-1),
            #  lat.unsqueeze(-1),
            #  days.unsqueeze(-1),
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
            compartment_embeddings.append(self.compartment_models[i](weighted_mean[:, i]).unsqueeze(1))

        graphs = torch.cat(compartment_embeddings, dim = 1) # shape (batch, n_compartments, n_embedding)


        # similarity = torch.bmm(self.key_linear(graphs), self.query_linear(graphs).permute(0, 2, 1)) #- torch.eye(self.n_compartments).to(self.device).unsqueeze(0) # [b, p, p]
        similarity = torch.bmm(self.E1.unsqueeze(0), self.E2.unsqueeze(0).permute(0,2,1)) #- torch.eye(self.n_compartments).to(self.device).unsqueeze(0) # [b, p, p]

        # Compute the magnitudes of each embedding vector
        magnitude = torch.norm(graphs, p= 2, dim= -1, keepdim=True) + 1e-10

        # Normalize the dot product by the magnitudes
        normalized_similarity_matrix = similarity / (magnitude * magnitude.transpose(1, 2))

        # if a row is all zeros, then the cosine similarity will be NaN, so we replace it with zeros
        cosine_similarities = normalized_similarity_matrix.mean(dim=0).abs()
        cosine_similarities[torch.isnan(cosine_similarities)] = 0

        
        
        def case_amplf_mask(attention, p=2.5, threshold=0.08):
            '''
            This function computes the case amplification mask for a 2D attention tensor 
            with the given amplification factor p.

            Parameters:
                - attention (torch.Tensor): A 2D attention tensor of shape [n, n].
                - p (float): The case amplification factor (default: 2.5).
                - threshold (float): The threshold for the mask (default: 0.05).

            Returns:
                - mask (torch.Tensor): A 2D binary mask of the same size as `attention`,
                where 0s denote noisy elements and 1s denote clean elements.
            '''
            # Compute the maximum value in the attention tensor
            max_val, _ = torch.max(attention.detach(), dim=1, keepdim=True)

            # Compute the mask as per the case amplification formula
            mask = (attention.detach() / max_val) ** p

            # Turn the mask into a binary matrix, where anything below threshold will be considered as zero
            mask = torch.where(mask > threshold, torch.tensor(1).to(attention.device), torch.tensor(0).to(attention.device))
            return mask
        
        mask = case_amplf_mask(cosine_similarities, p=2.5, threshold=0.18)

        cosine_similarities = cosine_similarities * mask

        assert not torch.isnan(graphs).any(), 'NaN in graphs'

        X_raw = graphs
        X_gnn = self.gcn(X_raw, cosine_similarities) # [b, p, e]

        X_raw = self.raw_map(X_raw) 
        X_hat = self.x_gnn_norm(X_gnn + X_raw)
        # X_hat = self.x_gnn_norm(X_gnn)
        X_hat = self.embedding_dropout(X_hat)   

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
    



class ProfileModelSUSTeR6(nn.Module):

    def __init__(self, n_features, n_compartments, n_embedding, n_dvdz_features, z_scale, device, profile_embedder = None, node_assigner = None) -> None:
        super(ProfileModelSUSTeR6, self).__init__()
        
        self.l1 = nn.Linear(n_features+12, 64)

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

        input = torch.cat(   [               
                argo_mean,
                fs.unsqueeze(-1),
                ac.unsqueeze(-1),
                ws,
            ],dim = -1)

        x = self.l1(input)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)

        x = self.l3(x)
        x = x.squeeze(dim = -1)

        return x, None


    
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