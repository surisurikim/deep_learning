"""
    @ author : Seul Kim
    @ when : Jan 01, 2024
    @ contact : niceonesuri@gmail.com
    @ blog : https://smartest-suri.tistory.com/
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(
            self, in_channels = 3, out_channels = 1, features = [64, 128, 256, 512]
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList() # Contracting path - 인코딩 파트의 모듈을 담을 리스트 선언
        self.ups = nn.ModuleList()   # Expanding path - 디코딩 파트의 모듈을 담을 리스트 선언
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2) # 풀링은 모든 블럭에서 공통 사용됨

        # Contracting path (Down - 인코딩 파트) ------------------------
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)) # 블록마다 더블콘볼루션 해주고 아웃풋은 feature맵 리스트 순서대로 할당(64, 128...)
            in_channels = feature # 다음 모듈의 인풋 사이즈를 feature로 업데이트

        # Bottleneck (인코딩, 디코딩 연결 파트) ---------------------------
        size = features[-1] # 512
        self.bottleneck = DoubleConv(size, size * 2) # 인풋 512 아웃풋 1024

        # Expanding path (Up - 디코딩 파트) ----------------------------
        for feature in features[::-1]: # 피처맵 사이즈 반대로!
            # 먼저 초록색 화살표에 해당하는 up-conv 레이어를 먼저 추가해 줍니다.
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size = 2, stride = 2
                    # 인풋에 *2 해주는 이유 : 나중에 skip-connection을 통해서 들어오는 인풋 사이즈가 더블이 되기 때문!
                    # kernel과 stride size가 2인 이유는.. 논문에서 그렇게 하겠다고 했음 '_^ 
                )
            )
            # 이제 더블 콘볼루션 레이어 추가
            self.ups.append(DoubleConv(feature * 2, feature))

        # Output (아웃풋 파트) -----------------------------------------
        last_input = features[0] # 64
        self.final_conv = nn.Conv2d(last_input, out_channels, kernel_size = 1)

    ####################### **-- forward pass --** #######################
    def forward(self, x):
        skip_connections = []

        # Contracting path (Down - 인코딩 파트) ------------------------
        for down in self.downs: # 인코딩 파트를 지나면서 각 블록에서 저장된 마지막 모듈 하나씩 이터레이션
            x = down(x)
            skip_connections.append(x) # skip_connection 리스트에 추가
            x = self.pool(x)

        # Bottleneck (인코딩, 디코딩 연결 파트) ---------------------------
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # 디코딩 파트에서 순서대로 하나씩 뽑기 편하게 리스트 순서 반대로 뒤집어주기

        # Expanding path (Up - 디코딩 파트) ----------------------------
        for i in range(len(skip_connections)):
            x = self.ups[i * 2](x) # self_ups에는 순서대로 ConvTranspose2d와 DoubleConv가 들어가 있음. 
            # 0, 2, 4... 짝수번째에 해당하는 인덱스만 지정하면 순서대로 ConvTranspose2d와(up-conv) 모듈만 지정하게 됨
            skip_connection = skip_connections[i] # skip_connection 순서대로 하나씩 뽑아서
            # concatenate 해서 붙여줄(connection) 차례!
            # 그런데 만약 붙일때 shape이 맞지 않는다면... (특히 이미지의 input_shape이 홀수인 경우 이런 뻑이 나게됨)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection.shape[2:]) # 간단히resize로 맞춰주겠음
            concat_skip = torch.cat((skip_connection, x), dim = 1) # 이제 붙임!
            x = self.ups[i * 2 + 1](concat_skip)
            # 1, 3, 5... 홀수번째에 해당하는 인덱스만 지정하면 순서대로 DoubleConv 모듈만 지정하게 됨
            
        return self.final_conv(x)
    
# 유넷 모델 테스트하는 함수 작성

def test():
    # 3장의 1채널(grayscale), width height가 각각 160인 랜덤 텐서 생성 (테스트용으로!)
    x = torch.randn((3, 1, 160, 160))
    # in_channels 1로 설정 (그레이스케일 이미지), out_channels 1로 설정 (binary output)
    model = UNET(in_channels = 1, out_channels = 1)
    # forward pass
    preds = model(x)
    # preds와 x의 shape 확인하기 - 두개가 같아야 함
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape 
        # True : 오케이
        # False : AssertionError

if __name__ == "__main__":
    test()
    