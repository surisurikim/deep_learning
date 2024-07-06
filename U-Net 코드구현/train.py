"""
    @ author : Seul Kim
    @ when : Jan 01, 2024
    @ contact : niceonesuri@gmail.com
    @ blog : https://smartest-suri.tistory.com/
"""

#%%

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.utils as vutils

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, datasets
from data import Mydataset # 내가 작성한 data.py에서 Mydataset 클래스 모듈 임포트

# ------------------------ data loader 정의하기 ------------------------ #
width = 360
height = 240
batch_size = 4
n_workers = 2

# 이미지만 정규화!
image_transforms = T.Compose([
    T.Resize((width, height)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 마스크는 정규화 안할거임
mask_transforms = T.Compose([
    T.Resize((width, height)),
    T.ToTensor()
])

train_dataset = Mydataset(root_dir = "flood/", 
                          train = True, # train 용 지정
                          image_transforms = image_transforms, 
                          mask_transforms = mask_transforms)

valid_dataset = Mydataset(root_dir = "flood/", 
                          train = False, # valid 용 지정 
                          image_transforms = image_transforms, 
                          mask_transforms = mask_transforms)

train_dataset_loader = DataLoader(dataset = train_dataset, 
                                  batch_size = batch_size, 
                                  shuffle = True, 
                                  num_workers = n_workers)

valid_dataset_loader = DataLoader(dataset = valid_dataset, 
                                  batch_size = batch_size, 
                                  shuffle = True, 
                                  num_workers = n_workers)

# ------------------------ Unet 모델 가져오기 ------------------------ #
from unet import UNET # 내가 작성한 unet.py 파일의 UNET 클래스 모듈 임포트 

# ------------------------- 평가지표 metrics ------------------------ #
# def dice_score(pred: torch.Tensor, mask: torch.Tensor): # 다이스 스코어
#     dice = (2 * (pred * mask).sum()) / (pred + mask).sum()
#     return np.mean(dice.cpu().numpy())


def dice_score(pred: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6):
    # 시그모이드 적용 후 임계값을 통해 이진화
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    # 마스크가 float 타입인지 확인
    mask = mask.float()
    
    # 교집합과 합집합 계산
    intersection = (pred * mask).sum(dim=[1, 2, 3])
    union = pred.sum(dim=[1, 2, 3]) + mask.sum(dim=[1, 2, 3])
    
    # Dice score 계산
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # 배치의 평균 Dice score 반환
    return dice.mean().item()

# dice score랑 loss 값만 확인하겠습니다..
# def iou_score(pred: torch.Tensor, mask: torch.Tensor):
#     # iou score...이건 나중에 (다이스만 보면 될거같음)
#     pass

# def pixel_accuracy(pred: torch.Tensor, mask: torch.Tensor): # 픽셀 어큐러시 (맞춘거/전체)
#     correct = torch.eq(pred, val_mask).int()
#     return float(correct.sum()) / float(correct.numel())

# ----------------------------- TRAIN ----------------------------- #

import numpy as np
from tqdm import tqdm 
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR, MultiStepLR, CyclicLR
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_iters = 10000
epochs = 10
learning_rate = 0.0002

# 그래프 그리기
def plot_pred_img(samples, pred):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))
    fig.tight_layout()

    ax1.axis('off')
    ax1.set_title('input image') # 원본 이미지 show
    ax1.imshow(np.transpose(vutils.make_grid(samples['img'], padding=2).numpy(),
                           (1, 2, 0)))

    ax2.axis('off')
    ax2.set_title('input mask') # 마스크 이미지 show
    ax2.imshow(np.transpose(vutils.make_grid(samples['mask'], padding=2).numpy(),
                           (1, 2, 0)), cmap='gray')
    
    ax3.axis('off')
    ax3.set_title('predicted mask') # 예측 이미지 show
    ax3.imshow(np.transpose(vutils.make_grid(pred, padding=2).cpu().numpy(),
                           (1, 2, 0)), cmap='gray')

    plt.show()
    
    
def plot_train_progress(model):
    model.eval() # 학습모드 아님 
    with torch.no_grad(): # gradient calculation 차단
        samples = next(iter(valid_dataset_loader))
        val_img = samples['img'].to(device)
        val_mask = samples['mask'].to(device)

        pred = model(val_img)
        plot_pred_img(samples, pred.detach())
    
    
########### --------- 본격 트레인 시작 --------- #############

def train(model, optimizer, criteration, scheduler=None):
    # 기록
    train_losses = []
    val_lossess = []
    lr_rates = []
    
    # train epochs 계산하기
    epochs = int(n_iters / (len(train_dataset) / batch_size))

    # epochs만큼 이터레이션
    for epoch in range(epochs):
        model.train() # ....................트레인모드 설정....................
        train_total_loss = 0
        train_iterations = 0
        
        for idx, data in enumerate(tqdm(train_dataset_loader)):
            train_iterations += 1
            train_img = data['img'].to(device)
            train_mask = data['mask'].to(device)
            
            optimizer.zero_grad() # 각 이터레이션 회차에서 grad를 0으로 초기화하는 역할 (이전 회차에서 쌓인 grad랑 분리 학습)
            with torch.autocast(device_type='cuda'): # training 스피드 업 (혼합 정밀도 학습 통해서 성능 향상, 메모리 사용량 감소)
                train_output_mask = model(train_img)
                train_loss = criterion(train_output_mask, train_mask)
                train_total_loss += train_loss.item()

            train_loss.backward() # # Backpropagate 통해서 gradient 누적 계산
            optimizer.step() # 모델 파라미터 update

        # loss 기록
        train_epoch_loss = train_total_loss / train_iterations
        train_losses.append(train_epoch_loss)
        
        model.eval() # ....................평가모드 설정....................
        with torch.no_grad(): # 평가(evaluation) 혹은 추론(inference)시 model.eval()과 함께 사용, 역전파를 통한 기울기 계산을 건너뛰어서 메모리 사용량을 줄이고 계산 속도를 높여줌
            val_total_loss = 0
            val_iterations = 0
            scores = 0

            # tqdm은 진행 상태(progress bar)를 시각적으로 보여주는 라이브러리
            for vidx, val_data in enumerate(tqdm(valid_dataset_loader)): # 인덱스 vidx와 실제 데이터 val_data를 동시에 제공
                val_iterations += 1
                val_img = val_data['img'].to(device)
                val_mask = val_data['mask'].to(device)

                with torch.autocast(device_type='cuda'):
                    pred = model(val_img) # 모델이 예측한 결과
                    val_loss = criterion(pred, val_mask) # 실제 마스크와 예측결과 사이의 loss 계산 -> criterion 은 나중에 BCEWithLogitsLoss()으로 할당될것
                    val_total_loss += val_loss.item() # val_loss가 PyTorch 텐서(tensor)형태로 반환됨(스칼라 단일값) -> Python의 기본 데이터 타입(float)으로 변환
                    scores += dice_score(pred, val_mask) # dice score 계산

            val_epoch_loss = val_total_loss / val_iterations
            dice_coef_scroe = scores / val_iterations

            val_lossess.append(val_epoch_loss)           

            plot_train_progress(model)
            print('epochs - {}/{} [{}/{}], dice score: {}, train loss: {}, val loss: {}'.format(
                epoch+1, epochs,
                idx+1, len(train_dataset_loader),
                dice_coef_scroe, train_epoch_loss, val_epoch_loss
            )) 
            
        lr_rates.append(optimizer.param_groups[0]['lr'])
        if scheduler:
            scheduler.step() # decay learning rate
            print('LR rate:', scheduler.get_last_lr())
            
    return {
        'lr': lr_rates,
        'train_loss': train_losses,
        'valid_loss': val_lossess
    }


model = UNET(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
history = train(model, optimizer, criterion)
# %%
