import torch
import torch.nn as nn
import numpy as np
from model.resnet import resnet18
from clip_modules.model_loader import load


class VPA(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.offset = offset
        dtype = None
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype

        self.clip_att, _ = load(config.clip_model, context_length=config.context_length)
        self.clip_obj, _ = load(config.clip_model, context_length=config.context_length)
        self.classifier_att = nn.Linear(768, len(classes))
        self.classifier_obj = nn.Linear(768, len(attributes))

        # self.encoder_obj = resnet18(pretrained=True)
        # self.encoder_obj.fc = nn.Linear(512, len(classes))

        # self.encoder_att = resnet18(pretrained=True)
        # self.encoder_att.fc = nn.Linear(512, len(attributes))


    def visual_obj(self, x: torch.Tensor):
        x = self.clip_obj.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_obj.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_obj.visual.positional_embedding.to(x.dtype)
        x = self.clip_obj.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip_obj.visual.transformer(x)

        x = img_feature.permute(1, 0, 2)  # LND -> NLD

        x = self.clip_obj.visual.ln_post(x[:, 0, :])
        if self.clip_obj.visual.proj is not None:
            x = x @ self.clip_obj.visual.proj
        return x
    
    def visual_att(self, x: torch.Tensor):
        x = self.clip_att.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_att.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_att.visual.positional_embedding.to(x.dtype)
        x = self.clip_att.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip_att.visual.transformer(x)

        x = img_feature.permute(1, 0, 2)  # LND -> NLD

        x = self.clip_att.visual.ln_post(x[:, 0, :])
        if self.clip_att.visual.proj is not None:
            x = x @ self.clip_att.visual.proj
        return x
    

    def forward(self, x):
        # logits_obj = self.encoder_obj(x)
        # logits_att = self.encoder_att(x)
        obj_feature = self.visual(x.type(self.clip_obj.dtype))   ## bs * 768
        att_feature = self.visual(x.type(self.clip_att.dtype))   ## bs * 768
        logits_att = self.classifier_att(att_feature)
        logits_obj = self.classifier_obj(obj_feature)

        return logits_att, logits_obj