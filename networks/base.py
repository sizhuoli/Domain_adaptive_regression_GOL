import ipdb
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_pretrained_vit import ViT


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.backbone == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            backbone.fc = nn.Identity()
            self.encoder = backbone

        elif cfg.backbone == 'vgg16':
            backbone = models.vgg16_bn(pretrained=True)
            backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            backbone.classifier = nn.Identity()
            self.encoder = backbone

        elif cfg.backbone == 'vgg16v2':  # no bn, relu, maxpool after last convolution
            backbone = models.vgg16_bn(pretrained=True)
            backbone.features[41] = nn.Identity()
            backbone.features[42] = nn.Identity()
            backbone.features[43] = nn.Identity()
            backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            backbone.classifier = nn.Identity()
            self.encoder = backbone

        elif cfg.backbone == 'vgg16v2norm':  # no bn, relu, maxpool after last convolution
            backbone = models.vgg16_bn(pretrained=True)
            backbone.features[41] = nn.Identity()
            backbone.features[42] = nn.Identity()
            backbone.features[43] = nn.Identity()
            backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            class Normalization(torch.nn.Module):
                """the resulting embedding is L2 normalized

                embedding = vector of length 512

                norm of embedding for each sample = 1

                """
                def __init__(self, dim=-1):
                    super().__init__()
                    self.dim = dim
                def forward(self, x):
                    return nn.functional.normalize(x, dim=self.dim)
            backbone.classifier = Normalization()
            self.encoder = backbone

        elif cfg.backbone == 'vgg16v2norm_reduce': # add one linear layer
            backbone = models.vgg16_bn(pretrained=True)
            backbone.features[41] = nn.Identity()
            backbone.features[42] = nn.Identity()
            backbone.features[43] = nn.Identity()
            backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            class Normalization(torch.nn.Module):
                """the resulting embedding is L2 normalized

                embedding = vector of length 512

                norm of embedding for each sample = 1

                """

                def __init__(self, dim=-1):
                    super().__init__()
                    self.dim = dim

                def forward(self, x):
                    return nn.functional.normalize(x, dim=self.dim)

            backbone.classifier = nn.Sequential(
                nn.Linear(512, cfg.backbone_end_dim),
                Normalization()
            )
            self.encoder = backbone


        elif cfg.backbone == 'vit_b16_reduce':  # add one linear layer
            class Normalization(torch.nn.Module):
                def __init__(self, dim=-1):
                    super().__init__()
                    self.dim = dim
                def forward(self, x):
                    return nn.functional.normalize(x, dim=self.dim)

            backbone = ViT('B_16_imagenet1k', pretrained=True, image_size=cfg.inputlength)
            backbone.fc = nn.Sequential(
                nn.Linear(768, cfg.backbone_end_dim),
                Normalization())
            self.encoder = backbone



        elif cfg.backbone == 'vgg19v2norm':  # no bn, relu, maxpool after last convolution
            backbone = models.vgg19_bn(pretrained=True)
            print('to be implemented')


        elif cfg.backbone == 'resnet34':
            class Normalization(torch.nn.Module):
                def __init__(self, dim=-1):
                    super().__init__()
                    self.dim = dim
                def forward(self, x):
                    return nn.functional.normalize(x, dim=self.dim)

            backbone = models.resnet34(pretrained=True)
            backbone.fc = Normalization()
            self.encoder = backbone



        elif cfg.backbone == 'resnet50':
            class Normalization(torch.nn.Module):
                def __init__(self, dim=-1):
                    super().__init__()
                    self.dim = dim
                def forward(self, x):
                    return nn.functional.normalize(x, dim=self.dim)
            backbone = models.resnet50(pretrained=True)
            backbone.fc = Normalization()
            # ipdb.set_trace()
            self.encoder = backbone


        elif cfg.backbone == 'vgg16fc':
            backbone = models.vgg16_bn(pretrained=True)
            backbone.classifier[5] = nn.Identity()
            backbone.classifier[6] = nn.Identity()
            self.encoder = backbone
        else:
            raise ValueError(f'Not supported backbone architecture {cfg.backbone}')

    def forward(self, x_base, x_ref=None):
        # feature extraction
        base_embs = self.encoder(x_base)
        if x_ref is not None:
            ref_embs = self.encoder(x_ref)
            out = self._forward(base_embs, ref_embs)
            return out, base_embs, ref_embs
        else:
            out = self._forward(base_embs)
            return out

    def _forward(self, base_embs, ref_embs=None):
        raise NotImplementedError('Suppose to be implemented by subclass')

