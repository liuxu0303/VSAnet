import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
import torchvision.transforms.transforms as trans
import numpy as np
from .ViT import ViT

config_dict = {
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
}



def std_preprocess(images, image_size):
    """(preprocessing)

    :param images: List[PIL.Image]
    :param image_size: int
    :return: shape(N, C, H, W)
    """
    output = []
    trans_func = trans.Compose([
        trans.Resize(image_size),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    for image in images:
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        output.append(trans_func(image))
    return torch.stack(output, dim=0)


def get_same_padding(in_size, kernel_size, stride):
    """'Same 'same' operation with tensorflow
    notice：padding=(0, 1, 0, 1) and padding=(1, 1, 1, 1) are different

    padding=(1, 1, 1, 1):
        out(H, W) = (in + [2 * padding] − kernel_size) // stride + 1

    'same' padding=(0, 1, 0, 1):
        out(H, W) = (in + [2 * padding] − kernel_size) / stride + 1

    :param in_size: Union[int, tuple(in_h, in_w)]
    :param kernel_size: Union[int, tuple(kernel_h, kernel_w)]
    :param stride: Union[int, tuple(stride_h, stride_w)]
    :return: padding: tuple(left, right, top, bottom)
    """
    in_h, in_w = (in_size, in_size) if isinstance(in_size, int) else in_size
    kernel_h, kernel_w = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    out_h, out_w = math.ceil(in_h / stride_h), math.ceil(in_w / stride_w)
    pad_h = max((out_h - 1) * stride_h + kernel_h - in_h, 0)
    pad_w = max((out_w - 1) * stride_w + kernel_w - in_w, 0)


    return pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2


def drop_connect(x, drop_p, training):
    """Throw away the whole InvertedResidual Module"""
    if not training:
        return x

    keep_p = 1 - drop_p
    keep_tensors = torch.floor(keep_p + torch.rand((x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device))
    return x / keep_p * keep_tensors


class Swish(nn.Module):
    class SwishImplement(Function):
        """output = x * sigmoid(x)

       
        Memory is more efficient, but the computing speed may be slightly reduced"""

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, output_grad):
            """d_output / dx = x * sigmoid'(x) + x' + sigmoid(x)"""
            x, = ctx.saved_tensors
            sigmoid_x = torch.sigmoid(x)
            return output_grad * (sigmoid_x * (x * (1 - sigmoid_x) + 1))

    def forward(self, x):
        return self.SwishImplement().apply(x)


class Conv2dStaticSamePadding(nn.Sequential):
    """Conv using 'same' padding in tensorflow"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, bias,
                 image_size):
        padding = get_same_padding(image_size, kernel_size, stride)
        super(Conv2dStaticSamePadding, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias)
        )


class Conv2dBNSwish(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, bias,
                 bn_momentum, bn_eps, image_size, norm_layer):
        super(Conv2dBNSwish, self).__init__(
            Conv2dStaticSamePadding(in_channels, out_channels, kernel_size, stride, groups, bias, image_size),
            norm_layer(out_channels, bn_eps, bn_momentum),
            Swish()
        )


class InvertedResidual(nn.Module):
    """Mobile Inverted Residual Bottleneck Block

    also be called MBConv or MBConvBlock"""

    def __init__(self, in_channels, out_channels, kernel_size, expand_ratio, id_skip, stride, se_ratio,
                 bn_momentum, bn_eps, image_size, drop_connect_rate, norm_layer):
        """
        Params:
            image_size(int): using static same padding, image_size is necessary.
            id_skip(bool): direct connection.
            se_ratio(float): reduce and expand
        """
        super(InvertedResidual, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.id_skip = id_skip
        self.drop_connect_rate = drop_connect_rate

        neck_channels = int(in_channels * expand_ratio)
        if expand_ratio > 1:
            self.expand_conv = Conv2dBNSwish(in_channels, neck_channels, 1, 1,
                                             1, False, bn_momentum, bn_eps, image_size, norm_layer)

        self.depthwise_conv = Conv2dBNSwish(neck_channels, neck_channels, kernel_size, stride,
                                            neck_channels, False, bn_momentum, bn_eps, image_size, norm_layer)
        if (se_ratio is not None) and (0 < se_ratio <= 1):
            se_channels = int(in_channels * se_ratio)
            # a Squeeze and Excitation layer
            self.squeeze_excitation = nn.Sequential(
                Conv2dStaticSamePadding(neck_channels, se_channels, 1, 1, 1, True, image_size),
                Swish(),
                Conv2dStaticSamePadding(se_channels, neck_channels, 1, 1, 1, True, image_size),
            )

        self.pointwise_conv = nn.Sequential(
            Conv2dStaticSamePadding(neck_channels, out_channels, 1, 1, 1, False, image_size),
            norm_layer(out_channels, bn_eps, bn_momentum),
        )

    def forward(self, inputs):
        x = inputs
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        if hasattr(self, 'squeeze_excitation'):
            z = torch.mean(x, dim=(2, 3), keepdim=True)  # AdaptiveAvgPool2d
            z = torch.sigmoid(self.squeeze_excitation(z))
            x = z * x  # se is like a door(sigmoid)
            del z
        x = self.pointwise_conv(x)

        if self.id_skip and self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = drop_connect(x, self.drop_connect_rate, training=self.training)  # x may be return zero
            x = x + inputs  # skip connection  if x == 0, x = inputs
        return x


class EfficientNet(nn.Module):
    def __init__(self,
                 width_ratio=1.0, depth_ratio=1.0, image_size=224, dropout_rate=0.2,
                 b0_inverted_residual_setting=None,
                 bn_momentum=1e-2, bn_eps=1e-3, channels_divisor=8, min_channels=None, drop_connect_rate=0.2,
                 norm_layer=None):
        super(EfficientNet, self).__init__()
        
        min_channels = min_channels or channels_divisor
        norm_layer = norm_layer or nn.BatchNorm2d

        if b0_inverted_residual_setting is None:
            # num_repeat, input_channels, output_channels will change.
            b0_inverted_residual_setting = [
                #  kernel_size(depthwise_conv), num_repeat, input_channels(first), output_channels(last),
                #  expand_ratio, .id_skip, stride(first), se_ratio
                [3, 1, 32, 16, 1, True, 1, 0.25],
                [3, 2, 16, 24, 6, True, 2, 0.25],
                [5, 2, 24, 40, 6, True, 2, 0.25],
                [3, 3, 40, 80, 6, True, 2, 0.25],
                [5, 3, 80, 112, 6, True, 1, 0.25],
                [5, 4, 112, 192, 6, True, 2, 0.25],
                [3, 1, 192, 320, 6, True, 1, 0.25]
            ]
        inverted_residual_setting = self._calculate_inverted_residual_setting(
            b0_inverted_residual_setting, width_ratio, depth_ratio, channels_divisor, min_channels)
        self.inverted_residual_setting = inverted_residual_setting
        # calculate total_block_num. Used to calculate the drop_connect_rate
        self.drop_connect_rate = drop_connect_rate
        self.block_idx = 0
        self.total_block_num = 0
        for setting in inverted_residual_setting:
            self.total_block_num += setting[1]

        # create modules
        out_channels = inverted_residual_setting[0][2]
        self.conv_first = Conv2dBNSwish(3, out_channels, 3, 2, 1, False, bn_momentum, bn_eps, image_size, norm_layer)
        for i, setting in enumerate(inverted_residual_setting):
            setattr(self, 'layer%d' % (i + 1), self._make_layers(setting, image_size, bn_momentum, bn_eps, norm_layer))

        in_channels = inverted_residual_setting[-1][3]
        out_channels = in_channels * 4
        self.conv_last = Conv2dBNSwish(in_channels, out_channels, 1, 1, 1, False,
                                       bn_momentum, bn_eps, image_size, norm_layer)

    def forward(self, x):
        x = self.conv_first(x)
        for i in range(len(self.inverted_residual_setting)):
            x = getattr(self, 'layer%d' % (i + 1))(x)
        x = self.conv_last(x) # The last convolution layer of efficientNet_B5 outputs the feature map F (n,C,H,W)   

        return x

    @staticmethod
    def _calculate_inverted_residual_setting(b0_inverted_residual_setting, width_ratio, depth_ratio,
                                             channels_divisor, min_channels):
        """change (num_repeat, input_channels, output_channels) through ratio"""
        inverted_residual_setting = b0_inverted_residual_setting.copy()
        for i in range(len(b0_inverted_residual_setting)):
            setting = inverted_residual_setting[i]
            # change input_channels, output_channels (width)  round
            setting[2], setting[3] = setting[2] * width_ratio, setting[3] * width_ratio
            in_channels, out_channels = \
                int(max(min_channels, int(setting[2] + channels_divisor / 2) // channels_divisor * channels_divisor)), \
                int(max(min_channels, int(setting[3] + channels_divisor / 2) // channels_divisor * channels_divisor))
            if in_channels < 0.9 * setting[2]:  # prevent rounding by more than 10%
                in_channels += channels_divisor
            if out_channels < 0.9 * setting[3]:
                out_channels += channels_divisor
            setting[2], setting[3] = in_channels, out_channels
            # change num_repeat (depth)  ceil
            setting[1] = int(math.ceil(setting[1] * depth_ratio))
        return inverted_residual_setting

    def _make_layers(self, setting, image_size, bn_momentum, bn_eps, norm_layer):
        """(Coupling) self.block_idx, self.total_block_num"""
        kernel_size, num_repeat, input_channels, output_channels, expand_ratio, id_skip, stride, se_ratio = setting
        layers = []
        for i in range(num_repeat):
            drop_connect_rate = self.drop_connect_rate * self.block_idx / self.total_block_num
            layers.append(InvertedResidual(
                input_channels if i == 0 else output_channels,
                output_channels, kernel_size, expand_ratio, id_skip,
                stride if i == 0 else 1,
                se_ratio, bn_momentum, bn_eps, image_size, drop_connect_rate, norm_layer
            ))
            self.block_idx += 1
        return nn.Sequential(*layers)

class VSAnet(nn.Module):
    def __init__(self, num_classes=9, n_bins=256, dropout_rate = 0.2, num_layers = 3, norm='linear', \
                 Query_list = [0.9, 3.3, 3.7, 0.26, 0.05, 0.05, 0.07, 0.0071, 0.173],\
                 norm_layer=None, **kwargs):  # Query_list is preset
        
        super(VSAnet, self).__init__()
        self.Query_list = Query_list
        self.cnn_encoder = EfficientNet(norm_layer=norm_layer, **kwargs) # CNN encoder
        self.dropout = nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=1) # E = 128
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0)
        self.transformer_encoder = ViT(128, n_query_channels=64, dim_out=n_bins,
                                        embedding_dim=128, num_layers=num_layers, norm=norm)
        
        self.fc = nn.Linear(2048, num_classes)
        
        self.softplus = nn.Softplus(beta=1, threshold=20)
        
    def forward(self, x):
        x = self.cnn_encoder(x)
        
        """classification module"""
        x = torch.mean(x, dim=(2, 3)) #mean pool
        x = self.dropout(x)
        cate_dist = self.fc(x)  # categorical distribution
        label  = F.gumbel_softmax(cate_dist, tau=0.1, hard = True)   #use Gumbel_Softmax, output One-hot Vector
        Query_list = np.array(self.Query_list)
        n, c = label.shape
        Query_list = np.tile(Query_list, (int(n),1)).astype(np.float32)
        Query_list = torch.Tensor(Query_list).to(x.device)  # shape = n, N
        areamax_pred = torch.sum(label * Query_list, dim=1, keepdim=True) # predicted A_max
        
        """area-bins module"""
        fuse_out = self.conv1(x)  #fuse channel information
        embeddings = self.conv2(fuse_out).flatten(2) 
        areabin_widths_normed, softmax_p = self.transformer_encoder(embeddings)  # output area-bin-weights b and Softmax score p
        areabin_widths =   areamax_pred * areabin_widths_normed  # .shape = N, dim_out
        areabin_widths = nn.functional.pad(areabin_widths, (1, 0), mode='constant', value= 0)
        areabin_edges = torch.cumsum(areabin_widths, dim=1)
        areabin_centers = 0.5 * (areabin_widths[:, :-1] + areabin_widths[:, 1:])  # caculate area-bin-centers
        
        n, dout = areabin_centers.size()
        areabin_centers = areabin_centers.view(n, dout)
        #predicted VSA caculated from the linear combination of p and area_bin_centers h(b)
        vsa_pred = torch.sum(softmax_p * areabin_centers, dim=1, keepdim=True) 
        
        return areabin_edges, vsa_pred, cate_dist, areamax_pred
    
    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.cnn_encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.fc, self.dropout, self.conv1, self.conv2, self.transformer_encoder]
        for m in modules:
            yield from m.parameters()
            
        

def _VSAestimator(model_name, pretrained=False, progress=True, num_classes=9,  n_bins=256, num_layers = 3, norm_layer=None, **kwargs):
    config = dict(zip(('width_ratio', 'depth_ratio', 'image_size', 'dropout_rate'), config_dict[model_name]))
    for key, value in config.items():
        kwargs.setdefault(key, value)

    model = VSAnet(num_classes, n_bins = n_bins, num_layers = num_layers, norm_layer=norm_layer, **kwargs)
    if pretrained:
        state_dict = torch.load('./models/path to pretrained checkpoints of efficientNer_B5')
        strict = True
        new_state_dict = {}
        if num_classes != 1000:
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")
            strict = False
        for k,v in state_dict.items():
            kk = 'cnn_encoder.' + k
            new_state_dict[kk] = v
        model.load_state_dict(new_state_dict, strict=strict)
    return model

def VSAestimator(pretrained=False, progress=True, num_classes=9,  n_bins=256, num_layers = 3, norm_layer=None, **kwargs):
    return _VSAestimator("efficientnet_b5", pretrained, progress, num_classes, n_bins, num_layers, norm_layer, **kwargs)

