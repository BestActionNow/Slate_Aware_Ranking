import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, emb_dim, feature_config, mlp_dims):
        super(DeepFM, self).__init__()
        # Wide part
        self.user_linear = nn.Embedding(feature_config['user_id'], 1)
        self.item_linear = nn.Embedding(feature_config['item_id'], 1)
        # Deep part
        self.user_emb_layer = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        input_dim = emb_dim * 2
        layers = list()
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def fm(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

    def forward(self, users, items, slate_ids=None, slate_poses=None, slate_ratings=None):
        uE = self.user_emb_layer(users.view(-1))
        uL = self.user_linear(users.view(-1))
        iE = self.item_emb_layer(items.view(-1))
        iL = self.item_linear(items.view(-1))
        bsz = uE.shape[0]
        dense_input = torch.stack([uE, iE], dim=1)
        mlp_term = self.mlp(dense_input.reshape([bsz, -1]))
        output = uL + iL + self.fm(dense_input) + mlp_term
        return torch.sigmoid(output), 0.1 * torch.mean(mlp_term ** 2)

class DeepFMslate(nn.Module):
    def __init__(self, emb_dim, feature_config, mlp_dims):
        super(DeepFMslate, self).__init__()
        # Wide part
        self.user_linear = nn.Embedding(feature_config['user_id'], 1)
        self.item_linear = nn.Embedding(feature_config['item_id'], 1)
        # Deep part
        self.user_emb_layer = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        input_dim = emb_dim * 3
        layers = list()
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        # Slate part
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

    def fm(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

    def forward(self, users, items, slate_ids=None, slate_poses=None, slate_ratings=None):
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
        
        fm_input = torch.stack([uE, iE], dim=1)
        dense_input = torch.cat([uE, iE, input_slate], dim=1)
        mlp_term = self.mlp(dense_input) 
        output = uL + iL + self.fm(fm_input) + mlp_term
        return torch.sigmoid(output), reg_loss + 0.1 * torch.mean(mlp_term ** 2)

class DeepFMPfd(nn.Module):
    def __init__(self, emb_dim, feature_config, mlp_dims):
        super(DeepFMPfd, self).__init__()
        self.user_linear = nn.Embedding(feature_config['user_id'], 1)
        self.item_linear = nn.Embedding(feature_config['item_id'], 1)
        self.user_emb_layer = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer = nn.Embedding(feature_config['item_id'], emb_dim)
        input_dim = emb_dim * 2
        layers = list()
        for embed_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self.user_linear_t = nn.Embedding(feature_config['user_id'], 1)
        self.item_linear_t = nn.Embedding(feature_config['item_id'], 1)
        self.user_emb_layer_t = nn.Embedding(feature_config['user_id'], emb_dim)
        self.item_emb_layer_t = nn.Embedding(feature_config['item_id'], emb_dim)
        input_dim = emb_dim * 3
        layers = list()
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

    def fm(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

    def forward(self, users, items, slate_ids=None, slate_poses=None, slate_ratings=None):
        uE = self.user_emb_layer(users.view(-1))
        uL = self.user_linear(users.view(-1))
        iE = self.item_emb_layer(items.view(-1))
        iL = self.item_linear(items.view(-1))
        bsz = uE.shape[0]
        dense_input = torch.stack([uE, iE], dim=1)
        output_s = uL + iL + self.mlp(dense_input.reshape([bsz, -1])) + self.fm(dense_input)

        uE = self.user_emb_layer_t(users.view(-1))
        uL = self.user_linear_t(users.view(-1))
        iE = self.item_emb_layer_t(items.view(-1))
        iL = self.item_linear_t(items.view(-1))
        iS = self.item_slate_emb_layer(slate_ids)
        targetS = self.item_slate_emb_layer(items)
        bsz = uE.shape[0]
        query = torch.stack(torch.split(self.att_q_trans(targetS), 4, dim=-1), 1) # b * 4 * 1 * 4
        key = torch.stack(torch.split(self.att_k_trans(iS), 4, dim=-1), 1) # b * 4 * 20 * 4
        value = torch.stack(torch.split(self.att_v_trans(iS), 4, dim=-1), 1) # b * 4 * 20 * 4
        weight = torch.matmul(query, torch.transpose(key, 2, 3)) / 2 # b * 4 * 1 * 20
        weight = torch.softmax(weight, dim=-1)
        iS_att = torch.matmul(weight, value).view([bsz, -1]) # b * 16
        dense_input = torch.stack([uE, iE, iS_att], dim=1)
        output_t = uL + iL + self.mlp_t(dense_input.reshape([bsz, -1])) + self.fm(dense_input)
        reg_loss = torch.square(output_s - output_t.detach()).sum(-1).mean()
        return torch.sigmoid(output_t), torch.sigmoid(output_s), reg_loss
