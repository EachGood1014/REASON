import torch
from torch import Tensor
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

__all__ = ['ResNet', 'De_ResNet', 'BN_layer', 'MultiProjectionLayer',
           'get_lns_layer', 'get_bn_layer', 'get_wide_resnet50_2', 'get_de_wide_resnet50_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

ckp_paths = {
    'resnet18': '/home/eachgood/Documents/AD_projects/REASON-main/pretrained_wigths/resnet18-f37072fd.pth',
    'resnet50': '/home/eachgood/Documents/AD_projects/REASON-main/pretrained_wigths/resnet50-0676ba61.pth',
    'wide_resnet50_2': '/home/eachgood/Documents/AD_projects/REASON-main/pretrained_wigths/wide_resnet50_2-95faca4d.pth'
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1,
              dilation: int = 1) -> nn.ConvTranspose2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class De_BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(De_BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if stride == 2:
            self.conv1 = deconv2x2(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            residual: bool = True,
            out_act: bool = False
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual
        self.out_act = out_act

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.residual:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

        out = self.relu(out)

        return out


class De_Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            residual: bool = True,
            out_act: bool = False
    ) -> None:
        super(De_Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            self.conv2 = deconv2x2(width, width, stride, groups, dilation)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual
        self.out_act = out_act

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.residual:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.noise_level = 0.1
        self.sample_noise_ratio = 0.5
        # self.channel_noise_ratio = 0.8
        self.channel_noise_ratio = 1.0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rand_seed = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def generate_noise(self, latent_feat: Tensor) -> Tensor:
        """
        Inject noise into the feature maps along channel dimension.
        """
        # random select sample from batch
        latent_noisy = latent_feat.clone()
        B, C, H, W = latent_noisy.shape
        num_noisy_samples = int(B * self.sample_noise_ratio)
        noisy_indices = torch.randperm(B)[:num_noisy_samples]

        num_noisy_channels = int(C * self.channel_noise_ratio)
        noisy_channel_indices = torch.randperm(C)[:num_noisy_channels]
        noise = torch.zeros((num_noisy_samples, C, H, W)).to(self.device)
        channel_noise = torch.normal(0, self.noise_std, (num_noisy_samples, num_noisy_channels, H, W)
                                     , generator=torch.Generator().manual_seed(self.rand_seed)) * self.noise_level

        noise[:, noisy_channel_indices] = channel_noise.to(self.device)
        self.rand_seed += 1
        latent_noisy[noisy_indices] += noise
        return latent_noisy

    def inject_local_noise(self, feature_maps, noise_ratio_range=(0.01, 0.5), noise_type='gaussian', noise_level=0.015,
                           num_positions=1):
        """
        Inject noise into local spatial positions of the feature maps.

        Args:
            feature_maps (torch.Tensor): The input feature maps of shape (B, C, H, W).
            noise_ratio_range (tuple): The ratio of the patch size to the feature map size.
            noise_type (str): The type of noise to inject ('gaussian' or 'uniform').
            noise_level (float): The standard deviation of the Gaussian noise or the range of the uniform noise.
            num_positions (int): The number of positions to inject noise.

        Returns:
            torch.Tensor: The feature maps with local noise injected.
        """
        B, C, H, W = feature_maps.shape

        noisy_feature_maps = feature_maps.clone()
        batch_indices = torch.randperm(B)[:B // 2]  # Randomly select half of the batch

        for _ in range(num_positions):
            noise_ratio = torch.FloatTensor(1).uniform_(*noise_ratio_range,
                                                        generator=torch.Generator().manual_seed(self.rand_seed)).item()
            # patch_size_h, patch_size_w = int(H * noise_ratio), int(W * noise_ratio)
            # random generate patch size and position
            # ToDo: add random patch_size_h and patch_size_w
            patch_size_h = torch.randint(1, max(int(H * noise_ratio), 2), (1,),
                                         generator=torch.Generator().manual_seed(self.rand_seed)).item()
            self.rand_seed += 1
            patch_size_w = torch.randint(1, max(int(W * noise_ratio), 2), (1,),
                                         generator=torch.Generator().manual_seed(self.rand_seed)).item()

            start_h = torch.randint(0, H - patch_size_h + 1, (1,),
                                    generator=torch.Generator().manual_seed(self.rand_seed)).item()
            start_w = torch.randint(0, W - patch_size_w + 1, (1,),
                                    generator=torch.Generator().manual_seed(self.rand_seed)).item()

            if noise_type == 'gaussian':
                noise = torch.randn((len(batch_indices), C, patch_size_h, patch_size_w),
                                    generator=torch.Generator().manual_seed(self.rand_seed)) * noise_level
            elif noise_type == 'uniform':
                noise = (torch.rand((len(batch_indices), C, patch_size_h, patch_size_w),
                                    generator=torch.Generator().manual_seed(self.rand_seed)) - 0.5) * 2 * noise_level
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")

            # Inject noise into the selected position
            noisy_feature_maps[batch_indices, :, start_h:start_h + patch_size_h,
            start_w:start_w + patch_size_w] += noise.to(self.device)

            self.rand_seed += 1
        return noisy_feature_maps

    def _forward_impl(self, x: Tensor, noise_injection: bool = False, noise_std: float = 0.015,
                      num_position: int = 1) -> [Tensor]:

        # x_noisy = None
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if noise_injection:
            # x_noisy = self.generate_noise(x)
            x_noisy = self.inject_local_noise(x, noise_ratio_range=(0.01, 0.5), noise_type='gaussian',
                                              noise_level=noise_std, num_positions=num_position)
        else:
            x_noisy = x

        feature_a = self.layer1(x_noisy)
        feature_b = self.layer2(feature_a)
        feature_c = self.layer3(feature_b)

        return [feature_a, feature_b, feature_c]

    def forward(self, x: Tensor, noise_injection: bool = False, noise_std: float = 0.015,
                num_position: int = 1) -> Tensor:
        return self._forward_impl(x, noise_injection, noise_std, num_position)


class De_ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, De_Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(De_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 512 * block.expansion
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 256, layers[0], stride=2, out_act=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], out_act=False)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], out_act=False)

        ###
        # self.deconv1 = deconv2x2(512 * block.expansion, 256 * block.expansion, 2)
        # self.bn1 = norm_layer(256 * block.expansion)
        # self.relu = nn.ReLU(inplace=True)
        # self.deconv2 = deconv2x2(256 * block.expansion, 128 * block.expansion, 2)
        # self.bn2 = norm_layer(128 * block.expansion)
        # self.deconv3 = deconv2x2(512 * block.expansion, 256 * block.expansion, 2)
        # self.bn3 = norm_layer(256 * block.expansion)
        ###

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, De_Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, De_Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, out_act: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> [Tensor]:

        feature_2 = self.layer1(x)
        # tmp_2 = self.relu(self.bn3(self.deconv3(x)))
        # feature_2 = feature_2 + tmp_2
        feature_1 = self.layer2(feature_2)

        # tmp_1 = self.relu(self.bn2(self.deconv2(self.relu(self.bn1(self.deconv1(x))))))
        # feature_1 = feature_1 + tmp_1
        feature_0 = self.layer3(feature_1)

        return [feature_0, feature_1, feature_2]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.load(ckp_paths[arch])
        model.load_state_dict(state_dict)
    return model


def _de_resnet(
        arch: str,
        block: Type[Union[De_BasicBlock, De_Bottleneck]],
        layers: List[int],
        **kwargs: Any
) -> De_ResNet:
    model = De_ResNet(block, layers, **kwargs)
    return model


class ProjLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(ProjLayer, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c // 2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c // 2, in_c // 4, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c // 4),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c // 4, in_c // 2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c // 2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c // 2, out_c, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(out_c),
                                  torch.nn.LeakyReLU(),
                                  )

    def forward(self, x):
        return self.proj(x)


class MultiProjectionLayer(nn.Module):
    def __init__(self, base=64):
        super(MultiProjectionLayer, self).__init__()
        self.proj_a = ProjLayer(base * 4, base * 4)
        self.proj_b = ProjLayer(base * 8, base * 8)
        self.proj_c = ProjLayer(base * 16, base * 16)

    def forward(self, features):
        return [self.proj_a(features[0]), self.proj_b(features[1]), self.proj_c(features[2])]


class DenoiseAE(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: int,
                 groups: int = 1,
                 width_per_group: int = 64,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
        super(DenoiseAE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1

        self.encoder = self._make_encoder_layer(block, 512, layers, stride=2, residual=True)
        self.decoder = self._make_decoder_layer(De_Bottleneck, 256, layers, stride=2, residual=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_encoder_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                            stride: int = 1, dilate: bool = False, residual=True) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes * 3, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, residual))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, residual=residual))

        return nn.Sequential(*layers)

    def _make_decoder_layer(self, block: Type[Union[BasicBlock, De_Bottleneck]], planes: int, blocks: int,
                            stride: int = 1, dilate: bool = False, residual=True) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion * 3, stride),
                norm_layer(planes * block.expansion * 3),
            )

        layers = []
        layers.append(block(self.inplanes, planes * 3, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, residual))
        self.inplanes = planes * block.expansion * 3
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes * 3, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, residual=residual))

        return nn.Sequential(*layers)

    def forward(self, x):
        hidden = self.encoder(x)
        return hidden, self.decoder(hidden)


class MMFusion(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]],
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MMFusion, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)

    def forward(self, x):
        h_1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        h_2 = self.relu(self.bn3(self.conv3(x[1])))
        h_3 = torch.cat([h_1, h_2, x[2]], 1)
        return h_3.contiguous()


class BN_layer(nn.Module):
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: int,
                 groups: int = 1,
                 width_per_group: int = 64,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(BN_layer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1

        # for multi-level native fusion
        self.mmf = MMFusion(block, norm_layer)

        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck, De_Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, residual=True) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes * 3, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: [Tensor]) -> [Tensor]:

        latent_feat = self.mmf(x)
        compact_feat = self.bn_layer(latent_feat)

        return compact_feat

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def get_wide_resnet50_2(pretrained: bool = False, progress: bool = True, noise_injection: bool = False,
                        **kwargs: Any) -> ResNet:
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def get_lns_layer(**kwargs: Any) -> MultiProjectionLayer:
    return MultiProjectionLayer()


def get_bn_layer(**kwargs: Any) -> BN_layer:
    kwargs['width_per_group'] = 64 * 2
    return BN_layer(Bottleneck, 3, **kwargs)


def get_de_wide_resnet50_2(pretrained: bool = False, progress: bool = True, s_arc: str = 't-like',
                           **kwargs: Any) -> De_ResNet:
    global layers
    kwargs['width_per_group'] = 64 * 2
    if s_arc == 't-anti':
        layers = [3, 6, 4, 3]
    elif s_arc == 't-like':
        layers = [3, 4, 6, 3]
    elif s_arc == 'plane':
        layers = [3, 3, 3, 3]
    return _de_resnet('wide_resnet50_2', De_Bottleneck, layers, **kwargs)


def get_resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def get_de_resnet50(pretrained: bool = False, **kwargs: Any) -> De_ResNet:
    return _de_resnet('resnet50', De_Bottleneck, [3, 4, 6, 3], **kwargs)


def get_bn_layer_resnet50(**kwargs: Any) -> BN_layer:
    return BN_layer(Bottleneck, 3, **kwargs)


def get_lns_layer_resnet50(**kwargs: Any) -> MultiProjectionLayer:
    return MultiProjectionLayer(base=64)

def get_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def get_de_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> De_ResNet:
    return _de_resnet('resnet18', De_BasicBlock, [2, 2, 2, 2], **kwargs)


def get_bn_layer_resnet18(**kwargs: Any) -> BN_layer:
    return BN_layer(BasicBlock, 3, **kwargs)


def get_lns_layer_resnet18(**kwargs: Any) -> MultiProjectionLayer:
    return MultiProjectionLayer(base=16)
