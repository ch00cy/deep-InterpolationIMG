import torch
import torch.nn as nn
import torchvision.transforms as transforms



class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution, Batch Normalization, ReLU 연산을 합친 함수
        def CBR2d(input_channel, output_channel, kernel_size=3, stride=1):
            layer = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm2d(num_features=output_channel),
                nn.ReLU()
            )
            return layer

        # Down Path ######################
        # Contracting path
        # conv 기본적으로 kernel size 3*3 에 stride 1으로 ■■■□□□ □■■■□□ □□■■■□ □□□■■■ =>2칸씩 크기가 줄어든다
        # 572x572x1 => 568x568x64
        self.conv1 = nn.Sequential(
            CBR2d(1, 64, 3, 1),
            CBR2d(64, 64, 3, 1)
        )
        # 568x568x64 => 284x284x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 284x284x64 => 280x280x128
        self.conv2 = nn.Sequential(
            CBR2d(64, 128, 3, 1),
            CBR2d(128, 128, 3, 1)
        )
        # 280x280x128 => 140x140x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 140x140x128 => 136x136x256
        self.conv3 = nn.Sequential(
            CBR2d(128, 256, 3, 1),
            CBR2d(256, 256, 3, 1)
        )
        # 136x136x256 => 68x68x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 68x68x256 => 64x64x512
        # Contracting path 마지막에 Dropout 적용
        self.conv4 = nn.Sequential(
            CBR2d(256, 512, 3, 1),
            CBR2d(512, 512, 3, 1),
            nn.Dropout(p=0.5)
        )
        # 64x64x512 => 32x32x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path 끝
        ###################################

        # Bottlneck 구간 ####################
        # 32x32x512 => 28x28x1024
        self.bottleNeck = nn.Sequential(
            CBR2d(512, 1024, 3, 1),
            CBR2d(1024, 1024, 3, 1),
        )
        # Bottlneck 구간 끝
        ###################################

        # Up Path #########################
        # Expanding path
        # channel 수를 감소 시키며 Up-Convolution
        # 28x28x1024 => 56x56x512
        self.upconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)

        # Up-Convolution 이후 channel = 512
        # Contracting path 중 같은 단계의 Feature map을 가져와 Up-Convolution 결과의 Feature map과 Concat 연산
        # => channel = 1024 가 됩니다.
        # forward 부분을 참고해주세요
        # 56x56x1024 => 52x52x512
        self.ex_conv1 = nn.Sequential(
            CBR2d(1024, 512, 3, 1),
            CBR2d(512, 512, 3, 1)
        )

        # 52x52x512 => 104x104x256
        self.upconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)

        # 104x104x512 => 100x100x256
        self.ex_conv2 = nn.Sequential(
            CBR2d(512, 256, 3, 1),
            CBR2d(256, 256, 3, 1)
        )

        # 100x100x256 => 200x200x128
        self.upconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        # 200x200x256 => 196x196x128
        self.ex_conv3 = nn.Sequential(
            CBR2d(256, 128, 3, 1),
            CBR2d(128, 128, 3, 1)
        )

        # 196x196x128 => 392x392x64
        self.upconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        # 392x392x128 => 388x388x64
        self.ex_conv4 = nn.Sequential(
            CBR2d(128, 64, 3, 1),
            CBR2d(64, 64, 3, 1),

        )

        # 논문 구조상 output = 2 channel
        # train 데이터에서 세포 / 배경을 검출하는것이 목표여서 class_num = 1로 지정
        # 388x388x64 => 388x388x1
        self.fc = nn.Conv2d(64, 1, kernel_size=1, stride=1)

    def forward(self, x):
        # Contracting path
        # 572x572x1 => 568x568x64
        layer1 = self.conv1(x)

        # Max Pooling
        # 568x568x64 => 284x284x64
        out = self.pool1(layer1)

        # 284x284x64 => 280x280x128
        layer2 = self.conv2(out)

        # Max Pooling
        # 280x280x128 => 140x140x128
        out = self.pool2(layer2)

        # 140x140x128 => 136x136x256
        layer3 = self.conv3(out)

        # Max Pooling
        # 136x136x256 => 68x68x256
        out = self.pool3(layer3)

        # 68x68x256 => 64x64x512
        layer4 = self.conv4(out)

        # Max Pooling
        # 64x64x512 => 32x32x512
        out = self.pool4(layer4)

        # bottleneck
        # 32x32x512 => 28x28x1024
        bottleNeck = self.bottleNeck(out)

        # Expanding path
        # 28x28x1024 => 56x56x512
        upconv1 = self.upconv1(bottleNeck)

        # Contracting path 중 같은 단계의 Feature map을 가져와 합침
        # Up-Convolution 결과의 Feature map size 만큼 CenterCrop 하여 Concat 연산
        # 56x56x512 => 56x56x1024
        cat1 = torch.cat((transforms.CenterCrop((upconv1.shape[2], upconv1.shape[3]))(layer4), upconv1), dim=1)
        # 레이어 4를 중간 기준으로 upconv1 의 h(upconv1.shape[2]),w(upconv1.shape[3]) 만큼 잘라서 □■ 나란히 연결

        # 56x56x1024 => 52x52x512
        ex_layer1 = self.ex_conv1(cat1)

        # 52x52x512 => 104x104x256
        upconv2 = self.upconv2(ex_layer1)

        # Contracting path 중 같은 단계의 Feature map을 가져와 합침
        # Up-Convolution 결과의 Feature map size 만큼 CenterCrop 하여 Concat 연산
        # 104x104x256 => 104x104x512
        cat2 = torch.cat((transforms.CenterCrop((upconv2.shape[2], upconv2.shape[3]))(layer3), upconv2), dim=1)
        # 레이어 3를 중간 기준으로 upconv2 의 h(upconv2.shape[2]),w(upconv2.shape[3]) 만큼 잘라서 □■ 나란히 연결

        # 104x104x512 => 100x100x256
        ex_layer2 = self.ex_conv2(cat2)

        # 100x100x256 => 200x200x128
        upconv3 = self.upconv3(ex_layer2)

        # Contracting path 중 같은 단계의 Feature map을 가져와 합침
        # Up-Convolution 결과의 Feature map size 만큼 CenterCrop 하여 Concat 연산
        # 200x200x128 => 200x200x256
        cat3 = torch.cat((transforms.CenterCrop((upconv3.shape[2], upconv3.shape[3]))(layer2), upconv3), dim=1)
        # 레이어 2를 중간 기준으로 upconv3 의 h(upconv3.shape[2]),w(upconv3.shape[3]) 만큼 잘라서 □■ 나란히 연결

        # 200x200x256 => 196x196x128
        ex_layer3 = self.ex_conv3(cat3)

        # 196x196x128=> 392x392x64
        upconv4 = self.upconv4(ex_layer3)

        # Contracting path 중 같은 단계의 Feature map을 가져와 합침
        # Up-Convolution 결과의 Feature map size 만큼 CenterCrop 하여 Concat 연산
        # 392x392x64 => 392x392x128
        cat4 = torch.cat((transforms.CenterCrop((upconv4.shape[2], upconv4.shape[3]))(layer1), upconv4), dim=1)
        # 레이어 1를 중간 기준으로 upconv4 의 h(upconv4.shape[2]),w(upconv4.shape[3]) 만큼 잘라서 □■ 나란히 연결

        # 392x392x128 => 388x388x64
        out = self.ex_conv4(cat4)

        # 388x388x64 => 388x388x1
        out = self.fc(out)
        return out