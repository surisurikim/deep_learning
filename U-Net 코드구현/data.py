from PIL import Image
from torch.utils.data import Dataset
import os

class Mydataset(Dataset):
    def __init__(self, root_dir='flood/', train=True, image_transforms=None, mask_transforms=None):
        super(Mydataset, self).__init__()
        self.train = train
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        
        # 파일 경로 지정
        file_path = os.path.join(root_dir, 'train')
        file_mask_path = os.path.join(root_dir, 'masked')
        
        # 이미지 리스트 생성
        self.images = sorted([os.path.join(file_path, img) for img in os.listdir(file_path)])
        self.masks = sorted([os.path.join(file_mask_path, mask) for mask in os.listdir(file_mask_path)])
        
        # train, valid 데이터 분리
        split_ratio = int(len(self.images) * 0.8)
        if train: 
            self.images = self.images[:split_ratio]
            self.masks = self.masks[:split_ratio]  # train은 80%
        else:
            self.images = self.images[split_ratio:]
            self.masks = self.masks[split_ratio:]  # valid는 20%
            
    def __getitem__(self, index: int):
        original = Image.open(self.images[index]).convert('RGB') # index 번째의 이미지를 RGB 형식으로 열음
        masked = Image.open(self.masks[index]).convert('L') # 얘는 마스크를 L(grayscale) 형식으로 열음
        
        if self.image_transforms:  # 나중에 image augmentation에 사용됨
            original = self.image_transforms(original)
        if self.mask_transforms:   # 나중에 image augmentation에 사용됨
            masked = self.mask_transforms(masked)
            
        return {'img': original, 'mask': masked} # transform이 적용된 후 텐서를 반환
    
    def __len__(self):
        return len(self.images)  # 이미지의 파일 수를 반환함 -> train = True라면 train 데이터셋의 크기를, False라면 valid 데이터셋의 크기를 반환
            
    
# ---------------------------- 데이터 확인하기 ----------------------------
import torch
from torchvision import transforms

# 이미지 텐서 확인
if __name__ == "__main__": # 현재 main 파일 안에서만 실행됨 (다른 파일에서 이 파일을 임포트 했을때는 실행되지 않아요)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 오리지날 이미지
    image_path = 'ASAC/suri/unets/flood/train/0.jpg' # 하나 예시 고름
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    print("이미지 shape : ", image_tensor.shape)
    print("이미지 텐서 값 확인 : ")
    print(image_tensor) 
    
    # 값의 범위가 [0, 1] 인가요?
    if torch.all(image_tensor <= 1.0):
        if torch.all(image_tensor >= 0.0):
         print("모든 픽셀의 값이 [0, 1] 범위를 가집니다")
    else:
        print("아니지롱")
        
    # 마스크
    mask_path = 'ASAC/suri/unets/flood/masked/0.png' # 하나 예시 고름
    mask = Image.open(mask_path).convert('RGB')
    mask_tensor = transform(mask)
    print("이미지 shape : ", mask_tensor.shape)
    print("이미지 텐서 값 확인 : ")
    print(mask_tensor)
    
    # 값의 범위가 [0, 1] 인가요?
    if torch.all(mask_tensor <= 1.0):
        if torch.all(mask_tensor >= 0.0):
         print("마스크도 모든 픽셀의 값이 [0, 1] 범위를 가집니다")
    else:
        print("아니지롱")

