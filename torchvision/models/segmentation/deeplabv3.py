from functools import partial
from typing import Any, List, Optional, OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from ...transforms._presets import SemanticSegmentation
from .._api import WeightsEnum, Weights
from .._meta import _VOC_CATEGORIES
from .._utils import IntermediateLayerGetter, handle_legacy_interface, _ovewrite_value_param
from ..mobilenetv3 import MobileNetV3, MobileNet_V3_Large_Weights, mobilenet_v3_large
from ..resnet import ResNet, resnet50, resnet101, resnet101_nch, ResNet50_Weights, ResNet101_Weights
from ._utils import _SimpleSegmentationModel
from .fcn import FCNHead


__all__ = [
    "DeepLabV3",
    "DeepLabV3_ResNet50_Weights",
    "DeepLabV3_ResNet101_Weights",
    "DeepLabV3_MobileNet_V3_Large_Weights",
    "deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass

class DeepLabV3Plus(_SimpleSegmentationModel):
    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result

class MultiTaskModel(nn.Module):
    def __init__(
        self, backbones: List[nn.Module], encoder_chs: List[int],
        hl_project: nn.Module, ll_project: nn.Module,
        height_decoder: nn.Module, seg_decoder: nn.Module, aux_classifier: Optional[nn.Module] = None
        ) -> None:
        super().__init__()
        self.backbones = backbones
        self.encoder_chs = encoder_chs
        self.hl_project = hl_project
        self.ll_project = ll_project
        self.height_decoder = height_decoder
        self.seg_decoder = seg_decoder
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        assert x.shape[1] == sum(self.encoder_chs), 'channels in input not equal to sum of encoders input channels!'
        # contract: features is a dict of tensors
        features_list = [
            backbone(
                x[:, sum(self.encoder_chs[:i]) : sum(self.encoder_chs[:i+1])]
            ) for i, backbone in enumerate(self.backbones) 
        ]
        features = {
            'low_level': self.ll_project(
                torch.cat([f['low_level'] for f in features_list], dim=1)
            ),
            'out': self.hl_project(
                torch.cat([f['out'] for f in features_list], dim=1)
            )
        }
        result = OrderedDict()
        height_x = self.height_decoder(features)
        seg_x = self.seg_decoder(features)
        height_x = F.interpolate(height_x, size=input_shape, mode="bilinear", align_corners=False)
        seg_x = F.interpolate(seg_x, size=input_shape, mode="bilinear", align_corners=False)
        result["height"] = height_x
        result["segmentation"] = seg_x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )

class DeepLabPlusHead(nn.Module):
    def __init__(self, in_channels: int, low_level_channels: int,  num_classes: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.aspp = ASPP(in_channels, [12, 24, 36])
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )
        self._init_weights()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        high_level_feature = self.aspp(feature['out'])
        upscale_hl_feature = F.interpolate(high_level_feature, size=low_level_feature.shape[-2:], mode='bilinear', align_corners=False)
        combined_feature = torch.cat([low_level_feature, upscale_hl_feature], dim=1)
        return self.classifier(combined_feature)

    def _init_weights(self): # need to do this because we have no pretraining for DeepLabV3+
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

class FeatureProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()
    
    def forward(self, x):
        return self.project(x)

    def _init_weights(self): # need to do this because we have no pretraining for DeepLabV3+
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _deeplabv3_resnet(
    backbone: ResNet,
    num_classes: int,
    aux: Optional[bool],
    plus: Optional[bool]=False
) -> DeepLabV3:
    return_layers = {"layer4": "out"}
    if plus:
        return_layers["layer1"] = "low_level"
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = DeepLabPlusHead(2048, 256, num_classes) if plus else DeepLabHead(2048, num_classes)

    if plus:
        return DeepLabV3Plus(backbone, classifier, aux_classifier)
    else:
        return DeepLabV3(backbone, classifier, aux_classifier)

def _multitask_model(
    backbones: List[nn.Module],
    encoder_chs: List[int],
    num_seg_classes: int,
    aux: Optional[bool]
) -> MultiTaskModel:
    return_layers = {"layer4": "out", "layer1": "low_level"}
    if aux:
        return_layers["layer3"] = "aux"
    backbones = [
        IntermediateLayerGetter(backbone, return_layers=return_layers) for backbone in backbones
    ]

    high_lvl_chs = 2048
    low_lvl_chs = 256
    high_lvl_project = FeatureProjection(high_lvl_chs*len(encoder_chs), high_lvl_chs)
    low_lvl_project = FeatureProjection(low_lvl_chs*len(encoder_chs), low_lvl_chs)

    aux_classifier = FCNHead(1024, num_seg_classes) if aux else None # kind of useless tbh
    height_decoder = DeepLabPlusHead(high_lvl_chs, low_lvl_chs, 1) # always one channel for height
    seg_decoder = DeepLabPlusHead(high_lvl_chs, low_lvl_chs, num_seg_classes)

    return MultiTaskModel(backbones, encoder_chs, high_lvl_project, low_lvl_project, height_decoder, seg_decoder, aux_classifier)

def multitask_model(
    *,
    weights: Optional[ResNet101_Weights] = None,
    progress: bool = True,
    num_seg_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet101_Weights] = ResNet101_Weights.IMAGENET1K_V2,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a multitask model with a ResNet-101 backbone and decoder heads for DSM height and surface segmentation.
    Based on DeepLabV3+

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.
    Reference: `Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation <https://arxiv.org/pdf/1802.02611>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_seg_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet101_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet101_Weights
        :members:
    """
    seg_weights = DeepLabV3_ResNet101_Weights.verify(weights)
    weights_backbone = ResNet101_Weights.verify(weights_backbone)

    if seg_weights is not None:
        weights_backbone = None
        num_seg_classes = _ovewrite_value_param(num_seg_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param(aux_loss, True)
    elif num_seg_classes is None:
        num_seg_classes = 21

    backbone = resnet101(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _multitask_model([backbone], [3], num_seg_classes, aux_loss)

    if seg_weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def multitask_model_multi_encoder(
    *,
    encoder_chs = [1],
    weights: Optional[ResNet101_Weights] = None,
    progress: bool = True,
    num_seg_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet101_Weights] = ResNet101_Weights.IMAGENET1K_V2,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a multitask model with a ResNet-101 backbone and decoder heads for DSM height and surface segmentation.
    Based on DeepLabV3+

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.
    Reference: `Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation <https://arxiv.org/pdf/1802.02611>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_seg_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet101_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet101_Weights
        :members:
    """
    seg_weights = DeepLabV3_ResNet101_Weights.verify(weights)
    # weights_backbone = ResNet101_Weights.verify(weights_backbone)
    weights_backbone = None # no pre-trained model available for single channel

    if seg_weights is not None:
        weights_backbone = None
        num_seg_classes = _ovewrite_value_param(num_seg_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param(aux_loss, True)
    elif num_seg_classes is None:
        num_seg_classes = 21

    backbones = [
        resnet101_nch(in_img_chs=chs, weights=weights_backbone, replace_stride_with_dilation=[False, True, True]) for chs in encoder_chs
    ]
    model = _multitask_model(backbones, encoder_chs, num_seg_classes, aux_loss)

    if seg_weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

_COMMON_META = {
    "categories": _VOC_CATEGORIES,
    "min_size": (1, 1),
}


class DeepLabV3_ResNet50_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 42004074,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_resnet50",
            "metrics": {
                "miou": 66.4,
                "pixel_acc": 92.4,
            },
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


class DeepLabV3_ResNet101_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 60996202,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#fcn_resnet101",
            "metrics": {
                "miou": 67.4,
                "pixel_acc": 92.4,
            },
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


class DeepLabV3_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_WITH_VOC_LABELS_V1 = Weights(
        url="https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
        transforms=partial(SemanticSegmentation, resize_size=520),
        meta={
            **_COMMON_META,
            "num_params": 11029328,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/segmentation#deeplabv3_mobilenet_v3_large",
            "metrics": {
                "miou": 60.3,
                "pixel_acc": 91.2,
            },
        },
    )
    DEFAULT = COCO_WITH_VOC_LABELS_V1


def _deeplabv3_mobilenetv3(
    backbone: MobileNetV3,
    num_classes: int,
    aux: Optional[bool],
) -> DeepLabV3:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    aux_inplanes = backbone[aux_pos].out_channels
    return_layers = {str(out_pos): "out"}
    if aux:
        return_layers[str(aux_pos)] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux else None
    classifier = DeepLabHead(out_inplanes, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


@handle_legacy_interface(
    weights=("pretrained", DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
def deeplabv3_resnet50(
    *,
    weights: Optional[DeepLabV3_ResNet50_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet50_Weights
        :members:
    """
    weights = DeepLabV3_ResNet50_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param(aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@handle_legacy_interface(
    weights=("pretrained", DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", ResNet101_Weights.IMAGENET1K_V1),
)
def deeplabv3_resnet101(
    *,
    weights: Optional[DeepLabV3_ResNet101_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet101_Weights] = ResNet101_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet101_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet101_Weights
        :members:
    """
    weights = DeepLabV3_ResNet101_Weights.verify(weights)
    weights_backbone = ResNet101_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param(aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet101(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def deeplabv3plus_resnet101(
    *,
    weights: Optional[DeepLabV3_ResNet101_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[ResNet101_Weights] = ResNet101_Weights.IMAGENET1K_V2,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet101_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet101_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet101_Weights
        :members:
    """
    weights = DeepLabV3_ResNet101_Weights.verify(weights)
    weights_backbone = ResNet101_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param(aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = resnet101(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss, plus=True)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


@handle_legacy_interface(
    weights=("pretrained", DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1),
    weights_backbone=("pretrained_backbone", MobileNet_V3_Large_Weights.IMAGENET1K_V1),
)
def deeplabv3_mobilenet_v3_large(
    *,
    weights: Optional[DeepLabV3_MobileNet_V3_Large_Weights] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.

    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.

    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained weights
            for the backbone
        **kwargs: unused

    .. autoclass:: torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights
        :members:
    """
    weights = DeepLabV3_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
        aux_loss = _ovewrite_value_param(aux_loss, True)
    elif num_classes is None:
        num_classes = 21

    backbone = mobilenet_v3_large(weights=weights_backbone, dilated=True)
    model = _deeplabv3_mobilenetv3(backbone, num_classes, aux_loss)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model
