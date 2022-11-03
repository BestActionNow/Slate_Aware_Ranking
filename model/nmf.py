import torch
import torch.nn as nn

class NMF(nn.Module):
    def __init__(self, emb_dim, feature_config, mlp_dims, out_alpha):
        super(NMF, self).__init__()
        self.user_emb_layer_G = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer_G = nn.Embedding(feature_config['item_id'], emb_dim)
        self.user_emb_layer = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        self.gmf = nn.Linear(emb_dim, 1)
        layers = list()
        input_dim = emb_dim * 2
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.out_alpha = out_alpha

    def forward(self, users, items, slate_ids=None, slate_poses=None, slate_ratings=None):
        uG = self.user_emb_layer_G(users.view(-1))
        iG = self.item_emb_layer_G(items.view(-1))
        uE = self.user_emb_layer(users.view(-1))
        iE = self.item_emb_layer(items.view(-1))
        output_gmf = self.gmf(torch.mul(uG, iG)) 
        dense_input = torch.cat([uE, iE], dim=1)
        output_mlp = self.mlp(dense_input)
        output = self.out_alpha * output_gmf + (1 - self.out_alpha) * output_mlp
        return torch.sigmoid(output), torch.mean(output_mlp ** 2)

class NMFSlate(torch.nn.Module):
    def __init__(self, emb_dim, feature_config, mlp_dims, out_alpha):
        super(NMFSlate, self).__init__()
        self.slate_size = feature_config['slate_size']
        self.user_emb_layer_G = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer_G = nn.Embedding(feature_config['item_id'], emb_dim)
        self.user_emb_layer = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        self.gmf = nn.Linear(emb_dim, 1)
        layers = list()
        input_dim = emb_dim * 3
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.out_alpha = out_alpha
        self.item_slate_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        self.encoder_student = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.encoder_teacher = nn.Sequential(
            nn.Linear(emb_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.att_q_trans = nn.Linear(emb_dim, 16)
        self.att_k_trans = nn.Linear(emb_dim, 16)
        self.att_v_trans = nn.Linear(emb_dim, 16)
        self.slate_trans = nn.Sequential(
            nn.Linear(emb_dim, 1)
        )

    def forward(self, users, items, slate_ids, slate_poses, slate_ratings):
        uG = self.user_emb_layer_G(users.view(-1))
        iG = self.item_emb_layer_G(items.view(-1))
        uE = self.user_emb_layer(users.view(-1))
        iE = self.item_emb_layer(items.view(-1))
        iS = self.item_slate_emb_layer(slate_ids)
        targetS = self.item_slate_emb_layer(items)
        bsz = uE.shape[0]
        output_student = self.encoder_student(uE)
        query = torch.stack(torch.split(self.att_q_trans(targetS), 4, dim=-1), 1) # b * 4 * 1 * 4
        key = torch.stack(torch.split(self.att_k_trans(iS), 4, dim=-1), 1) # b * 4 * 20 * 4
        value = torch.stack(torch.split(self.att_v_trans(iS), 4, dim=-1), 1) # b * 4 * 20 * 4
        weight = torch.matmul(query, torch.transpose(key, 2, 3)) / 2 # b * 4 * 1 * 20
        weight = torch.softmax(weight, dim=-1)
        iS_att = torch.matmul(weight, value).view([bsz, -1]) # b * 16
        input_teacher = torch.cat([uE, iS_att], dim=1)         
        output_teacher = self.encoder_teacher(input_teacher)
        reg_loss = torch.square(output_student - output_teacher).sum(-1).mean()
        if self.training:
            input_decoder = torch.cat([uE, output_teacher], 1)
        else:
            input_decoder = torch.cat([uE, output_student], 1)
        input_slate = self.decoder(input_decoder)
        output_gmf = self.gmf(torch.mul(uG, iG)) 
        dense_input = torch.cat([uE, iE, input_slate], dim=1)
        output_mlp = self.mlp(dense_input)
        output = self.out_alpha * output_gmf + (1 - self.out_alpha) * output_mlp
        # L2 NORM
        reg_loss += torch.mean(output_mlp ** 2)
        return torch.sigmoid(output), reg_loss

class NMFPfd(nn.Module):
    def __init__(self, emb_dim, feature_config, mlp_dims, out_alpha):
        super(NMFPfd, self).__init__()
        self.user_emb_layer_G = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer_G = nn.Embedding(feature_config['item_id'], emb_dim)
        self.user_emb_layer = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        self.gmf = nn.Linear(emb_dim, 1)
        layers = list()
        input_dim = emb_dim * 2
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.out_alpha = out_alpha

        self.user_emb_layer_G_t = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer_G_t = nn.Embedding(feature_config['item_id'], emb_dim)
        self.user_emb_layer_t = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer_t = nn.Embedding(feature_config['item_id'], emb_dim)
        self.gmf_t = nn.Linear(emb_dim, 1)
        layers = list()
        input_dim = emb_dim * 3
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp_t = nn.Sequential(*layers)

        self.item_slate_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        self.att_q_trans = nn.Linear(emb_dim, 16)
        self.att_k_trans = nn.Linear(emb_dim, 16)
        self.att_v_trans = nn.Linear(emb_dim, 16)
        self.slate_trans = nn.Sequential(
            nn.Linear(emb_dim, 1)
        )

    def forward(self, users, items, slate_ids=None, slate_poses=None, slate_ratings=None):
        uG = self.user_emb_layer_G(users.view(-1))
        iG = self.item_emb_layer_G(items.view(-1))
        uE = self.user_emb_layer(users.view(-1))
        iE = self.item_emb_layer(items.view(-1))
        output_gmf = self.gmf(torch.mul(uG, iG)) 
        dense_input = torch.cat([uE, iE], dim=1)
        output_mlp = self.mlp(dense_input)
        output_s = self.out_alpha * output_gmf + (1 - self.out_alpha) * output_mlp

        uG = self.user_emb_layer_G_t(users.view(-1))
        iG = self.item_emb_layer_G_t(items.view(-1))
        uE = self.user_emb_layer_t(users.view(-1))
        iE = self.item_emb_layer_t(items.view(-1))
        iS = self.item_slate_emb_layer(slate_ids)
        targetS = self.item_slate_emb_layer(items)
        bsz = uE.shape[0]
        query = torch.stack(torch.split(self.att_q_trans(targetS), 4, dim=-1), 1) # b * 4 * 1 * 4
        key = torch.stack(torch.split(self.att_k_trans(iS), 4, dim=-1), 1) # b * 4 * 20 * 4
        value = torch.stack(torch.split(self.att_v_trans(iS), 4, dim=-1), 1) # b * 4 * 20 * 4
        weight = torch.matmul(query, torch.transpose(key, 2, 3)) / 2 # b * 4 * 1 * 20
        weight = torch.softmax(weight, dim=-1)
        iS_att = torch.matmul(weight, value).view([bsz, -1]) # b * 16
        output_gmf = self.gmf_t(torch.mul(uG, iG)) 
        dense_input = torch.cat([uE, iE, iS_att], dim=1)
        output_mlp = self.mlp_t(dense_input)
        output_t = self.out_alpha * output_gmf + (1 - self.out_alpha) * output_mlp
        reg_loss = torch.square(output_s - output_t.detach()).sum(-1).mean()
        return torch.sigmoid(output_t), torch.sigmoid(output_s), reg_loss
