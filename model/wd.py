import torch
import torch.nn as nn

class WD(nn.Module):
    def __init__(self, emb_dim, feature_config, mlp_dims):
        super(WD, self).__init__()
        self.user_emb_layer = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        self.user_linear = nn.Embedding(feature_config['user_id'], 1)
        self.item_linear = nn.Embedding(feature_config['item_id'], 1)
        layers = list()
        input_dim = emb_dim * 2
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, users, items, slate_ids=None, slate_poses=None, slate_ratings=None):
        uE = self.user_emb_layer(users.view(-1))
        uL = self.user_linear(users.view(-1))
        iE = self.item_emb_layer(items.view(-1))
        iL = self.item_linear(items.view(-1))
        dense_input = torch.cat([uE, iE], dim=1)
        mlp_term = self.mlp(dense_input)
        output = uL + iL + mlp_term
        return torch.sigmoid(output), 0.1 * torch.mean(mlp_term ** 2)

class WDSlate(torch.nn.Module):
    def __init__(self, emb_dim, feature_config, mlp_dims):
        super(WDSlate, self).__init__()
        self.slate_size = feature_config['slate_size']
        self.user_emb_layer = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        self.user_linear = nn.Embedding(feature_config['user_id'], 1)
        self.item_linear = nn.Embedding(feature_config['item_id'], 1)
        layers = list()
        input_dim = emb_dim * 3
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
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
            nn.Linear(emb_dim + 16, emb_dim),
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
        uE = self.user_emb_layer(users.view(-1))
        uL = self.user_linear(users.view(-1))
        iE = self.item_emb_layer(items.view(-1))
        iL = self.item_linear(items.view(-1))
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
        dense_input = torch.cat([uE, iE, input_slate], dim=1)
        mlp_term = self.mlp(dense_input)
        output = uL + iL + mlp_term
        # L2 norm
        return torch.sigmoid(output), reg_loss + 0.1 * torch.mean(mlp_term ** 2)

class WDPfd(nn.Module):
    def __init__(self, emb_dim, feature_config, mlp_dims):
        super(WDPfd, self).__init__()
        self.user_emb_layer = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        self.user_linear = nn.Embedding(feature_config['user_id'], 1)
        self.item_linear = nn.Embedding(feature_config['item_id'], 1)
        layers = list()
        input_dim = emb_dim * 2
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self.user_emb_layer_t = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer_t = nn.Embedding(feature_config['item_id'], emb_dim)
        self.user_linear_t = nn.Embedding(feature_config['user_id'], 1)
        self.item_linear_t = nn.Embedding(feature_config['item_id'], 1)
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

    def forward(self, users, items, slate_ids=None, slate_poses=None, slate_ratings=None):
        # student
        uE = self.user_emb_layer(users.view(-1))
        uL = self.user_linear(users.view(-1))
        iE = self.item_emb_layer(items.view(-1))
        iL = self.item_linear(items.view(-1))
        dense_input = torch.cat([uE, iE], dim=1)
        mlp_term = self.mlp(dense_input)
        output_s = uL + iL + mlp_term
        # teacher
        uE_t = self.user_emb_layer_t(users.view(-1))
        uL_t = self.user_linear_t(users.view(-1))
        iE_t = self.item_emb_layer_t(items.view(-1))
        iL_t = self.item_linear_t(items.view(-1))
        iS = self.item_slate_emb_layer(slate_ids)
        targetS = self.item_slate_emb_layer(items)
        bsz = uE.shape[0]
        query = torch.stack(torch.split(self.att_q_trans(targetS), 4, dim=-1), 1) # b * 4 * 1 * 4
        key = torch.stack(torch.split(self.att_k_trans(iS), 4, dim=-1), 1) # b * 4 * 20 * 4
        value = torch.stack(torch.split(self.att_v_trans(iS), 4, dim=-1), 1) # b * 4 * 20 * 4
        weight = torch.matmul(query, torch.transpose(key, 2, 3)) / 2 # b * 4 * 1 * 20
        weight = torch.softmax(weight, dim=-1)
        iS_att = torch.matmul(weight, value).view([bsz, -1]) # b * 16
        dense_input = torch.cat([uE_t, iE_t, iS_att], dim=1)
        mlp_term_t = self.mlp_t(dense_input)
        output_t = uL_t + iL_t + mlp_term_t
        reg_loss = torch.square(output_s - output_t.detach()).sum(-1).mean()
        return torch.sigmoid(output_t), torch.sigmoid(output_s), 0.1 * reg_loss
