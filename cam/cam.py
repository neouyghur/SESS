import torch
import torch.nn.functional as F
from cam import BaseCAM

# class CAM(BaseCAM):
#     def __init__(self, model, target_layer=None):
#         super().__init__(model, target_layer)
#         self.weight_softmax = torch.squeeze(self.get_weight(model))
#
#     def get_weight(self, model):
#         params = list(model.parameters())
#         return params[-2]
#
#     def forward(self, x, class_idx=None, retain_graph=False):
#         x.requires_grad = False
#         b, c, h, w = x.size()
#
#         # predication on raw x
#         logit = F.softmax(self.model(x), dim=1)
#
#         if class_idx is None:
#             score = logit[:, logit.max(1)[-1]].squeeze()
#         else:
#             score = logit[:, class_idx].squeeze()
#
#         activations = self.activations['value'].data
#         b, k, u, v = activations.size()
#
#         weights = self.weight_softmax[class_idx].view(b, k, 1, 1).data
#         saliency_map = (weights * activations).sum(1, keepdim=True)
#
#         saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
#         saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
#         saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
#         return saliency_map.detach(), logit.detach()
#
#     def __call__(self, x, class_idx=None, retain_graph=False):
#         return self.forward(x, class_idx, retain_graph)

class CAM(BaseCAM):
    def __init__(self, model, target_layer=None):
        super().__init__(model, target_layer)
        self.weight_softmax = torch.squeeze(self.get_weight(model))

    def get_weight(self, model):
        params = list(model.parameters())
        return params[-2]

    def forward(self, x, class_idx=None, retain_graph=False):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
            x = x.to(next(self.model.parameters()).device)

        # x.requires_grad = False
        b, c, h, w = x.size()

        # predication on raw x
        logit = F.softmax(self.model(x), dim=1)

        if class_idx is None:
            class_idx = logit.max(1)[-1]

        activations = self.activations['value'].data
        b, k, u, v = activations.size()

        weights = self.weight_softmax[class_idx].view(1, k, 1, 1).data
        saliency_map = (weights * activations).sum(1, keepdim=True)

        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_shape = saliency_map.shape
        saliency_map = saliency_map.view(saliency_map.shape[0], -1)
        saliency_map_min, saliency_map_max = saliency_map.min(1, keepdim=True)[0], saliency_map.max(1, keepdim=True)[0]
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
        saliency_map = saliency_map.view(saliency_map_shape)
        return saliency_map.detach().cpu().numpy(), logit.detach()

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)