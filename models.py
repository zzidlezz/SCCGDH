import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0


    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        feat = self.alexnet.classifier(x)
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.fc3 = nn.Linear(txt_feat_len, code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = torch.tanh(self.alpha * hid)


        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class LabelNet(nn.Module):
    def __init__(self, label_dim,code_len):
        super(LabelNet, self).__init__()

        self.fc1 = nn.Linear(label_dim, (label_dim+code_len)//2)
        self.fc2 = nn.Linear((label_dim+code_len)//2, code_len)



    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code


    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)



