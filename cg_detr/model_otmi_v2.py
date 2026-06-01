import torch, torch.nn as nn, torch.nn.functional as F, sys
sys.path.insert(0, "/mnt/DATA/hari/vand/VTG_Project/CGDETR_OTMI")
from cg_detr.model_v2 import build_model as build_base
from cg_detr.matcher import build_matcher


class MIQueryInit(nn.Module):
    def __init__(self, d, nq=10):
        super().__init__()
        self.nq = nq
        self.gen = nn.Sequential(
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, nq * d)
        )
        self.orth = nn.Linear(d, d, bias=False)
        nn.init.orthogonal_(self.orth.weight)

    def mi_loss(self, q, vg):
        B, nq, D = q.shape
        ve  = vg.unsqueeze(1).expand(-1, nq, -1)
        pos = (q * ve).sum(-1).mean()
        idx = torch.randperm(B, device=q.device)
        neg = torch.log(torch.exp((q * ve[idx]).sum(-1)).mean() + 1e-6)
        return -(pos - neg)

    def div_loss(self, q):
        qn   = F.normalize(q, dim=-1)
        gram = torch.bmm(qn, qn.transpose(1,2))
        eye  = torch.eye(self.nq, device=q.device).unsqueeze(0)
        return ((gram - eye)**2).mean()

    def forward(self, vid, mask):
        B, T, D = vid.shape
        valid   = mask.float().unsqueeze(-1)
        vg      = (vid * valid).sum(1) / (valid.sum(1) + 1e-6)
        q       = self.orth(self.gen(vg).view(B, self.nq, D))
        return q, self.mi_loss(q, vg), self.div_loss(q)


class CGDETR_OTMI(nn.Module):
    """CG-DETR with OT transformer + MI query initialization"""
    def __init__(self, base_model, d, nq):
        super().__init__()
        self.base = base_model
        self.mi_query = MIQueryInit(d, nq=nq)
        self.lw_mi  = 0.1
        self.lw_div = 0.05

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask,
                vid=None, qid=None, src_aud=None, src_aud_mask=None,
                targets=None):
        # Get MI queries from video features
        B, T, _ = src_vid.shape
        # Use projected video as input to MI (simple linear proj)
        with torch.no_grad():
            vid_proj = self.base.input_vid_proj(src_vid)
        mi_q, l_mi, l_div = self.mi_query(vid_proj, src_vid_mask)

        # Run base CG-DETR forward (OT already in transformer)
        out = self.base(src_txt, src_txt_mask, src_vid, src_vid_mask,
                        vid=vid, qid=qid, targets=targets)

        # Add MI losses to output for criterion
        out['loss_mi']  = l_mi  * self.lw_mi
        out['loss_div'] = l_div * self.lw_div
        return out


def build_model(args):
    # Build base CG-DETR (OT already in transformer)
    base_model, criterion = build_base(args)

    d  = base_model.transformer.t2v_encoder.layers[0].self_attn.embed_dim
    nq = base_model.query_embed.weight.shape[0]

    model = CGDETR_OTMI(base_model, d, nq)
    return model, criterion
