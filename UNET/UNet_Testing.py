import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

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


class Dataset(torch.utils.data.Dataset):

    # torch.utils.data.Dataset 이라는 파이토치 base class를 상속받아
    # 그 method인 __len__(), __getitem__()을 오버라이딩 해줘서
    # 사용자 정의 Dataset class를 선언한다

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        # 문자열 검사해서 'label'이 있으면 True
        # 문자열 검사해서 'input'이 있으면 True
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    # 여기가 데이터 load하는 파트
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        inputs = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # normalize, 이미지는 0~255 값을 가지고 있어 이를 0~1사이로 scaling
        label = label / 255.0
        inputs = inputs / 255.0
        label = label.astype(np.float32)
        inputs = inputs.astype(np.float32)

        # 인풋 데이터 차원이 2이면, 채널 축을 추가해줘야한다.
        # 파이토치 인풋은 (batch, 채널, 행, 열)

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if inputs.ndim == 2:
            inputs = inputs[:, :, np.newaxis]

        data = {'input': inputs, 'label': label}

        if self.transform:
            data = self.transform(data)
        # transform에 할당된 class 들이 호출되면서 __call__ 함수 실행

        return data


class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # numpy와 tensor의 배열 차원 순서가 다르다.
        # numpy : (행, 열, 채널)
        # tensor : (채널, 행, 열)
        # 따라서 위 순서에 맞춰 transpose

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        # 이후 np를 tensor로 바꾸는 코드는 다음과 같이 간단하다.
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

## 하이퍼 파라미터 설정

lr = 1e-3
batch_size = 4
num_epoch = 100

data_dir = '/content/drive/My Drive/Colab Notebooks/파이토치/Architecture practice/UNet/data'
ckpt_dir = '/content/drive/My Drive/Colab Notebooks/파이토치/Architecture practice/UNet/checkpoint'
log_dir = '/content/drive/My Drive/Colab Notebooks/파이토치/Architecture practice/UNet/log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform 적용해서 데이터 셋 불러오기
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'),transform=transform)

# 불러온 데이터셋, 배치 size줘서 DataLoader 해주기
loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle=True)

# val set도 동일하게 진행
dataset_val = Dataset(data_dir=os.path.join(data_dir,'val'),transform = transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size , shuffle=True)

# 네트워크 불러오기
net = UNet().to(device) # device : cpu or gpu

# loss 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 정의
optim = torch.optim.Adam(net.parameters(), lr = lr )

# 기타 variables 설정
num_train = len(dataset_train)
num_val = len(dataset_val)

num_train_for_epoch = np.ceil(num_train/batch_size) # np.ceil : 소수점 반올림
num_val_for_epoch = np.ceil(num_val/batch_size)

# 기타 function 설정
fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0,2,3,1) # device 위에 올라간 텐서를 detach 한 뒤 numpy로 변환
fn_denorm = lambda x, mean, std : (x * std) + mean
fn_classifier = lambda x :  1.0 * (x > 0.5)  # threshold 0.5 기준으로 indicator function으로 classifier 구현

# Tensorbord
writer_train = SummaryWriter(log_dir=os.path.join(log_dir,'train'))
writer_val = SummaryWriter(log_dir = os.path.join(log_dir,'val'))


# 네트워크 저장하기
# train을 마친 네트워크 저장
# net : 네트워크 파라미터, optim  두개를 dict 형태로 저장
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, '%s/model_epoch%d.pth' % (ckpt_dir, epoch))


# 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):  # 저장된 네트워크가 없다면 인풋을 그대로 반환
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)  # ckpt_dir 아래 있는 모든 파일 리스트를 받아온다
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str, isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch


# 네트워크 학습시키기
start_epoch = 0
net, optim, start_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)  # 저장된 네트워크 불러오기

for epoch in range(start_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []

    for batch, data in enumerate(loader_train, 1):  # 1은 뭐니 > index start point
        # forward
        label = data['label'].to(device)  # 데이터 device로 올리기
        inputs = data['input'].to(device)
        output = net(inputs)

        # backward
        optim.zero_grad()  # gradient 초기화
        loss = fn_loss(output, label)  # output과 label 사이의 loss 계산
        loss.backward()  # gradient backpropagation
        optim.step()  # backpropa 된 gradient를 이용해서 각 layer의 parameters update

        # save loss
        loss_arr += [loss.item()]

        # tensorbord에 결과값들 저정하기
        label = fn_tonumpy(label)
        inputs = fn_tonumpy(fn_denorm(inputs, 0.5, 0.5))
        output = fn_tonumpy(fn_classifier(output))

        writer_train.add_image('label', label, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('input', inputs, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
        writer_train.add_image('output', output, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')

    writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

    # validation
    with torch.no_grad():  # validation 이기 때문에 backpropa 진행 x, 학습된 네트워크가 정답과 얼마나 가까운지 loss만 계산
        net.eval()  # 네트워크를 evaluation 용으로 선언
        loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward
            label = data['label'].to(device)
            inputs = data['input'].to(device)
            output = net(inputs)

            # loss
            loss = fn_loss(output, label)
            loss_arr += [loss.item()]
            print('valid : epoch %04d / %04d | Batch %04d \ %04d | Loss %04d' % (
            epoch, num_epoch, batch, num_val_for_epoch, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            inputs = fn_tonumpy(fn_denorm(inputs, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_classifier(output))

            writer_val.add_image('label', label, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('input', inputs, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('output', output, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        # epoch이 끝날때 마다 네트워크 저장
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

writer_train.close()
writer_val.close()