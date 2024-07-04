# https://huggingface.co/docs/transformers/v4.42.0/en/task_summary#depth-estimation

#%%
from transformers import pipeline
import torch

# 감정 분석 ---------------------------------------------------------------
classifier = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
test = ["I am very very hungry, do not piss me off.",
        "I miss her too much, it hurts.",
        "Are you serious? OMG! congrats!",
        "WTF?",
        "I"]

for sentence in test:
    result = classifier(sentence)
    print(sentence, "\n", result)
    
# 물건 탐지 --------------------------------------------------------------
import requests
from PIL import Image
from transformers import pipeline

url = "https://i.namu.wiki/i/--GbZ0ptaE0KF8OgUej9I_SN4erfOc_ueyHgtJipMB0scNAJRSio6uWMcFviEGKO0d0qSqwWhla7xGfiB5NYoQgAPSmh8TQW1AAuYljDuveZiAwd8kcbOV4mFFpCVz6CMZ9cBBym3rPK19df_Blbhw.webp"
image_data = requests.get(url, stream=True).raw # 이미지 데이터 가져오기
image = Image.open(image_data)

# 이미지 보기
import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis('off')
plt.show()

# 탐지
detector = pipeline(task = 'object-detection')
preds = detector(url)
preds = [{"score": round(pred["score"], 4), "label": pred["label"], "box": pred["box"]} for pred in preds]

print(f"총 {len(preds)}개의 객체가 탐지되었습니다!")
for pred in preds:
    print(pred)

# 박스 그리기
from matplotlib.patches import Rectangle

plt.figure(figsize = (15,8))
fig, ax = plt.subplots(1)
ax.imshow(image)
for pred in preds:
    box = pred["box"]
    label = pred["label"]
    score = pred["score"]
    xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
    width, height = xmax - xmin, ymax - ymin
    # 사각형 만들기
    rect = Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
    # 사각형 추가하기
    ax.add_patch(rect)
    # 레이블 추가하기
    plt.text(xmin, ymin, f'{label} ({score})', bbox=dict(facecolor='yellow', alpha=0.5))

plt.axis('off')
plt.show()

# score value Top 5개만 박스 그리기
preds = sorted(preds, key = lambda x: x['score'], reverse = True)[:5]
fig, ax = plt.subplots(1)
ax.imshow(image)
for pred in preds:
    box = pred["box"]
    label = pred["label"]
    score = pred["score"]
    xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
    width, height = xmax - xmin, ymax - ymin
    # 사각형 만들기
    rect = Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
    # 사각형 추가하기
    ax.add_patch(rect)
    # 레이블 추가하기
    plt.text(xmin, ymin, f'{label} ({score})', bbox=dict(facecolor='yellow', alpha=0.5))

plt.axis('off')
plt.show()

# 이번엔 겨울왕국

def object_detect_plot(url):
    image_data = requests.get(url, stream=True).raw # 이미지 데이터 가져오기
    image = Image.open(image_data)
    detector = pipeline(task = 'object-detection')
    preds = detector(url)
    preds = [{"score": round(pred["score"], 4), "label": pred["label"], "box": pred["box"]} for pred in preds]

    print(f"총 {len(preds)}개의 객체가 탐지되었습니다!")
    for pred in preds:
        print(pred)
    
    plt.figure(figsize = (15,8))
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for pred in preds:
        box = pred["box"]
        label = pred["label"]
        score = pred["score"]
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        width, height = xmax - xmin, ymax - ymin
        # 사각형 만들기
        rect = Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        # 사각형 추가하기
        ax.add_patch(rect)
        # 레이블 추가하기
        plt.text(xmin, ymin, f'{label} ({score})', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()

# 함수화
url = "https://i.namu.wiki/i/0rikeSGdRdb6anXM7uHfm-NBtV4FGcEFkhqjj5Ilb3BbNUZDXPpIWTdMJ1bi8h1PLoRJn82VrVC2JZpf5c90_E8srgC2J3ZoeZ5UdMHBgauiAYMDtBT_lMGbuzIjv_cgqdzWFA5d9EER7frogh3FcA.webp"
object_detect_plot(url)

# 치맥
chicken_n_beer = "https://i.namu.wiki/i/2JQMZZIxjIeZpag74qgmIQvBrS9gcBy-w_iTkHgQ34V8pS63SaWqUTgnMZGxJykuwBdXXPLUr6IRv7jCsLnQlVI-t6L37ZTo3CLlGIaCjDnnThCMtCzm4l1QjC2wLva-mkj4CqNtE716a1mERKcn5A.webp"
object_detect_plot(chicken_n_beer)


# 깊이 추정 --------------------------------------------------------------
depth_estimator = pipeline(task="depth-estimation")
img = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image_data = requests.get(img, stream=True).raw # 이미지 데이터 가져오기
image = Image.open(image_data)

plt.imshow(image)
plt.axis(False)
plt.show()

preds = depth_estimator(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
# 결과 딕셔너리
preds

# 시각화
depth_image = preds['depth']
plt.imshow(depth_image, cmap='viridis')
plt.axis('off')
plt.colorbar()
plt.show()

# %%

