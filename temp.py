import torch

ckpt = torch.load("checkpoints/ovseg_R101c_vitB16_ft_mpt.pth.pt", map_location="cuda")
print(ckpt)
sd = ckpt["model"]  # または ckpt["state_dict"]

mask_emb = sd["clip_adapter.clip_model.visual.mask_embedding"]
print(mask_emb.shape)  # torch.Size([3, 196, 768])
