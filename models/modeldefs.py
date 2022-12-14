import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
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

    def forward(self, x):
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


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out = self.layer3(x)
        x = self.layer4(out)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, out

    def forward(self, x):
        y_logits, features = self._forward_impl(x)
        return y_logits, F.avg_pool2d(features, features.shape[2]).flatten(1)


def _resnet(arch, block, layers, ckpt_path=None):
    model = ResNet(block, layers)
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
        print('Loaded Imagenet checkpoint')
    return model


def resnet18(ckpt_path):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], ckpt_path)


def resnet50(ckpt_path):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], ckpt_path)



class MLPLayer(nn.Module):
    def __init__(self, activation, input_dim, output_dim):
        super().__init__()

        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        elif activation == 'identity':
            self.activation_fn = nn.Identity()
        else:
            raise NotImplementedError("Only 'relu', 'sigmoid' and 'identity' activations are supported")

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.activation_fn(self.linear(x))
        return x

class MLP(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, activation, final_activation):
        super(MLP, self).__init__()

        layers = [MLPLayer(activation, input_dim, hidden_dim)]

        for i in range(1, n_layers - 1):
            layers.append(MLPLayer(activation, hidden_dim, hidden_dim))

        layers.append(MLPLayer(final_activation, hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self,x):
        out = self.mlp(x)
        return out

def decoder(n_layers=5, input_dim=512, hidden_dim=512): #n_layers=1, input_dim=256, hidden_dim=512
    output_dim = input_dim // 2
    model = MLP(n_layers, input_dim, hidden_dim, output_dim, 'relu', 'identity')
    return model

def correct_incorrect(n_layers=3, input_dim=256, hidden_dim=512): #n_layers=3, input_dim=256, hidden_dim=512
    output_dim = 1
    model = MLP(n_layers, input_dim, hidden_dim, output_dim, 'relu', 'identity')
    return model

def error_estimator(n_layers=3, input_dim=256, hidden_dim=512):
    output_dim = 1
    model = MLP(n_layers, input_dim, hidden_dim, output_dim, 'relu', 'identity')
    return model


class linear_regressor(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super(linear_regressor, self).__init__()
        hidden_dim = 30
        layers = [MLPLayer('relu', input_dim, hidden_dim)]
        #layers.append(MLPLayer('relu', hidden_dim, hidden_dim))
        layers.append(MLPLayer('identity', hidden_dim, output_dim))
        #layers.append(MLPLayer(activation, hidden_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self,x):
        out = self.mlp(x)
        return out


class DeltaEnc():
    def __init__(self,
                 network,
                 optimizer,
                 X_train,
                 y_train,
                 ):

        self.f_predictor = network
        self.f_optimizer = optimizer

        self.X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
        self.y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
        self.input_dim = X_train.shape[1]
        self.output_dim = y_train.shape[1]

        self.loss_fn = nn.MSELoss()
        self.epoch = 0
        self.device = device

    def fit(self, epochs=500):
        data = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(data, shuffle=True, batch_size=100)

        self.f_predictor.train()
        for epoch in range(epochs):
            avg_loss = 0.0
            for batch_id, (xi, yi) in enumerate(loader):
                xi = xi.to(self.device)
                yi = yi.to(self.device)
                print(xi.shape, yi.shape)

                flipped_x = torch.flip(xi,[0])
                diff_x = xi-flipped_x
                inp = torch.cat([flipped_x,diff_x],axis=1)
                #flipped_y = torch.flip(yi,[0])
                #diff_y = yi-flipped_y
                out = yi

                out_hat = self.f_predictor(inp)
                self.f_optimizer.zero_grad()
                f_loss = self.loss_fn(out_hat, out)
                f_loss.backward()
                self.f_optimizer.step()
                avg_loss += f_loss.detach().item()/len(loader)

            self.epoch += 1
            print('Epoch {} Avg.train loss {}'.format(epoch, avg_loss))
            print('R2 = {}'.format(r2_score(out.cpu().detach().numpy().ravel(), out_hat.cpu().detach().numpy().ravel())))

    def save_ckpt(self, path, name):
        state = {'epoch': self.epoch}
        state['state_dict'] = self.f_predictor.state_dict()
        filename = path + '/' + name
        torch.save(state, filename)
        print('Saved Ckpts')

    def load_ckpt(self, path, name):
        self.f_predictor.load_state_dict(torch.load(os.path.join(path, name))['state_dict'])
        print('Loaded Ckpts')

    def _map_delta_model(self,ref,query):
        diff = query-ref
        samps = torch.cat([ref,diff],1)
        pred = self.f_predictor(samps)
        return pred

    def get_prediction_with_uncertainty(self, q):
        nref=np.minimum(20, self.X_train.shape[0])

        all_preds = []
        n_test = q.shape[0]
        for i in list(np.random.choice(self.X_train.shape[0], nref)):
            ref = self.X_train[i].to(self.device)
            ref_y = self.y_train[i].to(self.device)
            all_preds.append(self._map_delta_model(ref.expand([n_test,ref.shape[0]]),q.float()))

        all_preds = torch.stack(all_preds).squeeze(2)
        mu = torch.mean(all_preds,axis=0)
        var = torch.var(all_preds,axis=0)

        #all_preds = torch.stack(all_preds).view(n_test,nref,1)
        #mu = torch.mean(all_preds,axis=1)
        #var = torch.var(all_preds,axis=1)

        return mu,var


    def get_uncertainties_prediction(self, X_test):
        X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
        self.f_predictor.eval()

        X_test = X_test.to(self.device)
        mean, var = self.get_prediction_with_uncertainty(X_test)
        return mean, var
