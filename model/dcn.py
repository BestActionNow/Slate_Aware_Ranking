import torch
import torch.nn as nn

class DCN(nn.Module):
    def __init__(self, emb_dim, feature_config, dcn_layer_num, mlp_dims):
        super(DCN, self).__init__()
        self.user_emb_layer = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        # DCN part
        self.dcn_layer_num = dcn_layer_num
        dcn_input_dim = 2 * emb_dim
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(dcn_input_dim, 1, bias=False) for _ in range(dcn_layer_num)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((dcn_input_dim,))) for _ in range(dcn_layer_num)
        ])
        # Deep part
        layers = list()
        input_dim = 2 * emb_dim
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        self.mlp = nn.Sequential(*layers)
        # Last layer part
        self.linear = torch.nn.Linear(mlp_dims[-1] + dcn_input_dim, 1)


    def cross(self, x):
        x0 = x
        for i in range(self.dcn_layer_num):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x

    def forward(self, users, items, slate_ids=None, slate_poses=None, slate_ratings=None):
        uE = self.user_emb_layer(users.view(-1))
        iE = self.item_emb_layer(items.view(-1))
        dense_input = torch.concat([uE, iE], dim=1)
        x_l1 = self.cross(dense_input)
        h_l2 = self.mlp(dense_input)
        x_concat = torch.concat([x_l1, h_l2], dim=1)
        output = self.linear(x_concat)
        return torch.sigmoid(output), 0.1 * torch.mean(output ** 2)

class DCNSlate(nn.Module):
    def __init__(self, emb_dim, feature_config, dcn_layer_num, mlp_dims):
        super(DCNSlate, self).__init__()
        self.user_emb_layer = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        # DCN part
        self.dcn_layer_num = dcn_layer_num
        dcn_input_dim = 3 * emb_dim
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(dcn_input_dim, 1, bias=False) for _ in range(dcn_layer_num)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((dcn_input_dim,))) for _ in range(dcn_layer_num)
        ])
        # Deep part
        layers = list()
        input_dim = 3 * emb_dim
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        self.mlp = nn.Sequential(*layers)
        # Last layer part
        self.linear = torch.nn.Linear(mlp_dims[-1] + dcn_input_dim, 1)

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

    def cross(self, x):
        x0 = x
        for i in range(self.dcn_layer_num):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x

    def forward(self, users, items, slate_ids=None, slate_poses=None, slate_ratings=None):
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

        dense_input = torch.concat([uE, iE, input_slate], dim=1)
        x_l1 = self.cross(dense_input)
        h_l2 = self.mlp(dense_input)
        x_concat = torch.concat([x_l1, h_l2], dim=1)
        output = self.linear(x_concat)
        return torch.sigmoid(output), reg_loss + 0.1 * torch.mean(torch.sum(h_l2, dim=1) ** 2) \
            + 0.1 * torch.mean(torch.sum(x_l1, dim=1) ** 2)

class DCNPfd(nn.Module):
    def __init__(self, emb_dim, feature_config, dcn_layer_num, mlp_dims):
        super(DCNPfd, self).__init__()
        self.user_emb_layer = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        self.item_slate_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        self.att_q_trans = nn.Linear(emb_dim, 16)
        self.att_k_trans = nn.Linear(emb_dim, 16)
        self.att_v_trans = nn.Linear(emb_dim, 16)
        # DCN part
        self.dcn_layer_num = dcn_layer_num
        dcn_input_dim = 2 * emb_dim
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(dcn_input_dim, 1, bias=False) for _ in range(dcn_layer_num)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((dcn_input_dim,))) for _ in range(dcn_layer_num)
        ])
        # Deep part
        layers = list()
        input_dim = 2 * emb_dim
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        self.mlp = nn.Sequential(*layers)
        # Last layer part
        self.linear = torch.nn.Linear(mlp_dims[-1] + dcn_input_dim, 1)

        self.user_emb_layer_t = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer_t = nn.Embedding(feature_config['item_id'], emb_dim)
        # DCN part
        dcn_input_dim = 2 * emb_dim
        self.w_t = torch.nn.ModuleList([
            torch.nn.Linear(dcn_input_dim, 1, bias=False) for _ in range(dcn_layer_num)
        ])
        self.b_t = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((dcn_input_dim,))) for _ in range(dcn_layer_num)
        ])
        # Deep part
        layers = list()
        input_dim = 2 * emb_dim
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        self.mlp_t = nn.Sequential(*layers)
        # Last layer part
        self.linear_t = torch.nn.Linear(mlp_dims[-1] + dcn_input_dim + emb_dim, 1)

    def cross(self, x):
        x0 = x
        for i in range(self.dcn_layer_num):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x

    def cross_t(self, x):
        x0 = x
        for i in range(self.dcn_layer_num):
            xw = self.w_t[i](x)
            x = x0 * xw + self.b_t[i] + x
        return x

    def forward(self, users, items, slate_ids=None, slate_poses=None, slate_ratings=None):
        # student
        uE = self.user_emb_layer(users.view(-1))
        iE = self.item_emb_layer(items.view(-1))
        dense_input = torch.concat([uE, iE], dim=1)
        x_l1 = self.cross(dense_input)
        h_l2 = self.mlp(dense_input)
        x_concat = torch.concat([x_l1, h_l2], dim=1)
        output_s = self.linear(x_concat)
        # teacher
        iS = self.item_slate_emb_layer(slate_ids)
        targetS = self.item_slate_emb_layer(items)
        bsz = uE.shape[0]
        query = torch.stack(torch.split(self.att_q_trans(targetS), 4, dim=-1), 1) # b * 4 * 1 * 4
        key = torch.stack(torch.split(self.att_k_trans(iS), 4, dim=-1), 1) # b * 4 * 20 * 4
        value = torch.stack(torch.split(self.att_v_trans(iS), 4, dim=-1), 1) # b * 4 * 20 * 4
        weight = torch.matmul(query, torch.transpose(key, 2, 3)) / 2 # b * 4 * 1 * 20
        weight = torch.softmax(weight, dim=-1)
        iS_att = torch.matmul(weight, value).view([bsz, -1]) # b * 16
        uE = self.user_emb_layer_t(users.view(-1))
        iE = self.item_emb_layer_t(items.view(-1))
        dense_input = torch.concat([uE, iE], dim=1)
        x_l1 = self.cross_t(dense_input)
        h_l2 = self.mlp_t(dense_input)
        x_concat = torch.concat([x_l1, h_l2, iS_att], dim=1)
        output_t = self.linear_t(x_concat)
        # pfd reg loss
        reg_loss = torch.square(output_s - output_t.detach()).sum(-1).mean()
        return torch.sigmoid(output_t), torch.sigmoid(output_s), reg_loss
