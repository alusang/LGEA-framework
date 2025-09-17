import types
import torch
import transformers
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import pdb
import math
from .NewEA_tools import MultiModalEncoder
from .NewEA_loss import CustomMultiLossLayer, icl_loss, ial_loss
from .NewEA_tools import MultiModalFusion

from src.utils import pairwise_distances
import os.path as osp
import json
import pickle

class CrossAttention(nn.Module):
    def __init__(self, args, head=2):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.head = head
        self.h_size = self.hidden_size // self.head
        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_output = nn.Linear(self.hidden_size, self.hidden_size)
        self.param_init()
        
    def param_init(self):
        nn.init.xavier_normal_(self.linear_q.weight)
        nn.init.xavier_normal_(self.linear_k.weight)
        nn.init.xavier_normal_(self.linear_v.weight)
        nn.init.xavier_normal_(self.linear_output.weight)

    def calculate(self, Q, K, V, mask):
        attn = torch.matmul(Q, torch.transpose(K,-1,-2))
        if mask is not None: attn = attn.masked_fill(mask, -1e9)
        attn = torch.softmax(attn / (Q.size(-1) ** 0.5), dim=-1)
        attn = torch.matmul(attn, V)
        return attn

    def forward(self, x, y, attention_mask=None):
        batch_size = x.size(0)
        q_s = self.linear_q(x).view(batch_size, -1, self.head, self.h_size).transpose(1, 2)
        k_s = self.linear_k(y).view(batch_size, -1, self.head, self.h_size).transpose(1, 2)
        v_s = self.linear_v(y).view(batch_size, -1, self.head, self.h_size).transpose(1, 2)
        if attention_mask is not None: attention_mask = attention_mask.eq(0)
        attn = self.calculate(q_s, k_s, v_s, attention_mask)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, self.hidden_size)
        attn = self.linear_output(attn)
        return attn


class NewEA(nn.Module):
    def __init__(self, kgs, args):
        super().__init__()
        self.kgs = kgs
        self.args = args
        self.img_features = F.normalize(torch.FloatTensor(kgs["images_list"])).cuda()
        self.input_idx = kgs["input_idx"].cuda()
        self.adj = kgs["adj"].cuda()
        self.rel_features = torch.Tensor(kgs["rel_features"]).cuda()
        self.att_features = torch.Tensor(kgs["att_features"]).cuda()
        self.name_features = None
        self.char_features = None

        if kgs["name_features"] is not None:
            self.name_features = kgs["name_features"].cuda()
            self.char_features = kgs["char_features"].cuda()

        img_dim = self._get_img_dim(kgs)

        self.mask_a = kgs['mask_a']

        ent2id_dict = {}
        with open(self.args.ent2id_path, 'r', encoding='utf-8') as f:
            for line in f:
                ent_id, name = line.strip().split('\t')
                ent2id_dict[int(ent_id)] = name
        ent_num = len(ent2id_dict)
        mask_i = self.get_mask(ent2id_dict, ent_num, self.args.mask_path)
        mask_i = mask_i.clone().detach().cuda()
        self.mask_i = mask_i

        self.multimodal_encoder = MultiModalEncoder(args=self.args,
                                                    kgs=self.kgs,
                                                    ent_num=kgs["ent_num"],
                                                    img_feature_dim=img_dim,
                                                    char_feature_dim=None,
                                                    attr_input_dim=kgs["att_features"].shape[1],mask_i=self.mask_i)

        self.multi_loss_layer = CustomMultiLossLayer(loss_num=6)
        self.criterion_cl = icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)

        self.idx_one = torch.ones(self.args.batch_size, dtype=torch.int64).cuda()
        self.idx_double = torch.cat([self.idx_one, self.idx_one]).cuda()
        self.last_num = 1000000000000

        self.criterion_align = ial_loss(tau=self.args.tau2,
                                        ab_weight=self.args.ab_weight,
                                        zoom=self.args.zoom,
                                        reduction=self.args.reduction)
        self.align_multi_loss_layer = CustomMultiLossLayer(loss_num=6)  # 6
        
    def generate_hidden_emb(self, hidden):
        gph_emb = F.normalize(hidden[:, 0, :].squeeze(1))
        rel_emb = F.normalize(hidden[:, 1, :].squeeze(1))
        att_emb = F.normalize(hidden[:, 2, :].squeeze(1))
        img_emb = F.normalize(hidden[:, 3, :].squeeze(1))
        if hidden.shape[1] >= 6:
            name_emb = F.normalize(hidden[:, 4, :].squeeze(1))
            char_emb = F.normalize(hidden[:, 5, :].squeeze(1))
            joint_emb = torch.cat([gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb], dim=1)
        else:
            name_emb, char_emb = None, None
            loss_name, loss_char = None, None
            joint_emb = torch.cat([gph_emb, rel_emb, att_emb, img_emb], dim=1)

        return gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, joint_emb
        

    def forward(self, batch):
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb = self.joint_emb_generat(only_joint=False)
        
        loss_joi = self.criterion_cl(joint_emb, batch)

        in_loss = self.inner_view_loss(gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch)

        align_loss = self.kl_alignment_loss(joint_emb, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, batch)

        loss_all = loss_joi + in_loss + align_loss

        loss_dic = {"joint_Intra_modal": loss_joi.item(), "Intra_modal": in_loss.item(), "Inter_modal": align_loss.item()}
        output = {"loss_dic": loss_dic, "emb": joint_emb}
        return loss_all, output

    def miss_generation(self, e_r, e_i, e_a, a_mask, i_mask):
        with torch.no_grad():
            mask_row = a_mask * i_mask
            e_i_detach = e_i.detach()
            e_a_detach = e_a.detach()
            e_r_detach = e_r.detach()

        ir_input = self.fc_map_1(torch.cat((e_i_detach, e_r_detach), dim=-1))
        ar_input = self.fc_map_2(torch.cat((e_a_detach, e_r_detach), dim=-1))

        gen_ir, ir_mu, ir_logvar, ir_latent = self.ir_vae(ir_input)
        gen_a, a_mu, a_logvar, a_latent = self.a_vae(e_a_detach)

        comp_a = self.a_vae.decode(ir_latent)

        gen_ar, ar_mu, ar_logvar, ar_latent = self.ar_vae(ar_input)
        gen_i, i_mu, i_logvar, i_latent = self.i_vae(e_i_detach)

        comp_i = self.i_vae.decode(ar_latent)

        mmd_loss = self.gene_loss(gen_a[mask_row.bool()], e_a_detach[mask_row.bool()], a_mu[mask_row.bool()],
                                  a_logvar[mask_row.bool()]) + self.gene_loss(gen_ir[mask_row.bool()],
                                                                              ir_input[mask_row.bool()],
                                                                              ir_mu[mask_row.bool()], ir_logvar[
                                                                                  mask_row.bool()]) + self.gene_loss(
            gen_i[mask_row.bool()], e_i_detach[mask_row.bool()], i_mu[mask_row.bool()],
            i_logvar[mask_row.bool()]) + self.gene_loss(gen_ar[mask_row.bool()], ar_input[mask_row.bool()],
                                                        ar_mu[mask_row.bool()],
                                                        ar_logvar[mask_row.bool()]) + 0.01 * nn.MSELoss()(
            a_latent[mask_row.bool()], ir_latent[mask_row.bool()]) + 0.01 * nn.MSELoss()(
            i_latent[mask_row.bool()], ar_latent[mask_row.bool()])

        e_i_comp = torch.where(i_mask.unsqueeze(-1).bool(), e_i, comp_i)
        e_a_comp = torch.where(a_mask.unsqueeze(-1).bool(), e_a, comp_a)
        return mmd_loss, e_a_comp, e_i_comp

    def gene_loss(self, recon_output, original_output, mu, logvar):
        recon_loss = nn.MSELoss()(recon_output, original_output)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def r_rep(self, e):
        return F.normalize(self.ent_embed(e), 2, -1)

    def i_rep(self, e):
        return F.normalize(self.fc_i(self.img_embed(e)), 2, -1)

    def a_rep(self, e):
        return F.normalize(self.fc_a(self.atr_embed(e)), 2, -1)
        
    def get_score(self, emb, train_links, emb2=None, norm=True):
        if norm:
            emb = F.normalize(emb, dim=1)
            if emb2 is not None:
                emb2 = F.normalize(emb2, dim=1)
        num_ent = emb.shape[0]
        # Get (normalized) hidden1 and hidden2.
        zis = emb[train_links[:, 0]]
        if emb2 is not None:
            zjs = emb2[train_links[:, 1]]
        else:
            zjs = emb[train_links[:, 1]]
        score = torch.mm(zis, zjs.t())
        return score
        
    def inner_view_loss(self, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill):
        loss_GCN = self.criterion_cl(gph_emb, train_ill) if gph_emb is not None else 0
        loss_rel = self.criterion_cl(rel_emb, train_ill) if rel_emb is not None else 0
        loss_att = self.criterion_cl(att_emb, train_ill) if att_emb is not None else 0
        loss_img = self.criterion_cl(img_emb, train_ill) if img_emb is not None else 0
        loss_name = self.criterion_cl(name_emb, train_ill) if name_emb is not None else 0
        loss_char = self.criterion_cl(char_emb, train_ill) if char_emb is not None else 0

        total_loss = self.multi_loss_layer([loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char])
        return total_loss

    def kl_alignment_loss(self, joint_emb, gph_emb, rel_emb, att_emb, img_emb, name_emb, char_emb, train_ill):

        zoom = self.args.zoom
        loss_GCN = self.criterion_align(gph_emb, joint_emb, train_ill) if gph_emb is not None else 0
        loss_rel = self.criterion_align(rel_emb, joint_emb, train_ill) if rel_emb is not None else 0
        loss_att = self.criterion_align(att_emb, joint_emb, train_ill) if att_emb is not None else 0
        loss_img = self.criterion_align(img_emb, joint_emb, train_ill) if img_emb is not None else 0
        loss_name = self.criterion_align(name_emb, joint_emb, train_ill) if name_emb is not None else 0
        loss_char = self.criterion_align(char_emb, joint_emb, train_ill) if char_emb is not None else 0

        total_loss = self.align_multi_loss_layer(
            [loss_GCN, loss_rel, loss_att, loss_img, loss_name, loss_char]) * zoom
        return total_loss

    def joint_emb_generat(self, only_joint=True):
        gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb = self.multimodal_encoder(self.input_idx,
                                                                    self.adj,
                                                                    self.img_features,
                                                                    self.rel_features,
                                                                    self.att_features,
                                                                    self.name_features,
                                                                    self.char_features)
        if only_joint:
            return joint_emb
        else:
            return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb

    def _get_img_dim(self, kgs):
        if isinstance(kgs["images_list"], list):
            img_dim = kgs["images_list"][0].shape[1]
        elif isinstance(kgs["images_list"], np.ndarray) or torch.is_tensor(kgs["images_list"]):
            img_dim = kgs["images_list"].shape[1]
        return img_dim

    def Iter_new_links(self, epoch, left_non_train, final_emb, right_non_train, new_links=[]):
        if len(left_non_train) == 0 or len(right_non_train) == 0:
            return new_links
        distance_list = []
        for i in np.arange(0, len(left_non_train), 1000):
            d = pairwise_distances(final_emb[left_non_train[i:i + 1000]], final_emb[right_non_train])
            distance_list.append(d)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance, final_emb
        if (epoch + 1) % (self.args.semi_learn_step * 5) == self.args.semi_learn_step:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if preds_r[p] == i]
        else:
            new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l) if
                         (preds_r[p] == i) and ((left_non_train[i], right_non_train[p]) in new_links)]

        return new_links

    def get_mask(self, ent2id_dict, ent_num, file_path=None):
        if file_path is None:
            mask = torch.ones(ent_num, dtype=torch.float32)
        else:
            mask = torch.zeros(ent_num, dtype=torch.float32)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                id1 = int(item["id1"])
                id2 = int(item["id2"])
                value = item["mask"]
                if id1 in ent2id_dict:
                    mask[id1] = 1 if value > 0 else 0
                if id2 in ent2id_dict:
                    mask[id2] = 1 if value > 0 else 0
        return mask

    def cross_attention(self, a, b, c):
        w_normalized = F.softmax(self.modal_weight, dim=-1)
        ab, ac = self.ca_ab(b, a), self.ca_ac(c, a)
        a_align = w_normalized[1, 0] * a + w_normalized[1, 1] * ab + w_normalized[1, 2] * ac
        ba, bc = self.ca_ba(a, b), self.ca_bc(c, b)
        b_align = w_normalized[2, 0] * b + w_normalized[2, 1] * ba + w_normalized[2, 2] * bc
        ca, cb = self.ca_ca(a, c), self.ca_cb(b, c)
        c_align = w_normalized[3, 0] * c + w_normalized[3, 1] * ca + w_normalized[3, 2] * cb
        joint_emb = torch.cat([a_align, b_align, c_align], dim=1)

        orth_loss = self.orth_loss(b, a-ab) + self.orth_loss(c, a-ac) + self.orth_loss(a, b-ba) + self.orth_loss(c, b-bc) + self.orth_loss(a, c-ca) + self.orth_loss(b, c-cb)
        orth_loss = 0.01 * orth_loss

        return joint_emb, orth_loss

    def orth_loss(self, x, y):
        orth = torch.mean(x * y, dim=-1)
        loss = torch.mean(torch.pow(orth, 2))
        return loss

    def data_refresh(self, logger, train_ill, test_ill_, left_non_train, right_non_train, new_links=[]):
        if len(new_links) != 0 and (len(left_non_train) != 0 and len(right_non_train) != 0):
            new_links_select = new_links
            train_ill = np.vstack((train_ill, np.array(new_links_select)))
            num_true = len([nl for nl in new_links_select if nl in test_ill_])
            # remove from left/right_non_train
            for nl in new_links_select:
                left_non_train.remove(nl[0])
                right_non_train.remove(nl[1])

            if self.args.rank == 0:
                logger.info(f"#new_links_select:{len(new_links_select)}")
                logger.info(f"train_ill.shape:{train_ill.shape}")
                logger.info(f"#true_links: {num_true}")
                logger.info(f"true link ratio: {(100 * num_true / len(new_links_select)):.1f}%")
                logger.info(f"#entity not in train set: {len(left_non_train)} (left) {len(right_non_train)} (right)")

            new_links = []
        else:
            logger.info("len(new_links) is 0")

        return left_non_train, right_non_train, train_ill, new_links