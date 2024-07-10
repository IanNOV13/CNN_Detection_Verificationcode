import numpy as np
from keras.models import load_model
from callbacks import weighted_categorical_crossentropy
import cv2

#設定基本資料位置
numpy_save_folder = './numpy_data/2/'
model_name = 'best_model_bfloat16.h5'

#載入權重資料
class_weights = np.load(numpy_save_folder + 'class_weights.npy')

print("".join("*" for _ in range(10)))
for i,data in enumerate(class_weights):
    print("第%s權重資料:\n%s"%(i+1,data))
print("".join("*" for _ in range(10)))

#設定loss參數
loss=[
        weighted_categorical_crossentropy(class_weights[0]),
        weighted_categorical_crossentropy(class_weights[1]),
        weighted_categorical_crossentropy(class_weights[2]),
        weighted_categorical_crossentropy(class_weights[3])
]

#設定輸出字符
characters = 'abcdefghijklmnopqrstuvwxyz'
num_classes = len(characters)
dict_labels = {idx: char for idx, char in enumerate(characters)}


#嘗試加載模型
try:
    model = load_model(model_name,custom_objects={'loss': loss})  # 尝试加载最佳权重文件
    print("成功加载模型权重。")
except Exception as e:
    print(f"加载模型权重失败: {str(e)}")

#迴圈判斷驗證碼
while True:
    path = ".\\train_dataset\\2\\" + str(input('驗證碼名稱:')) + ".png"
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    test_feature_vector = img.reshape(
        1,100,120,1
    ).astype('float32')

    test_feature_normalize = test_feature_vector / 255

    predictions = model.predict(test_feature_normalize)

    character = ''
    for i, prediction in enumerate(predictions):
        character += dict_labels[np.argmax(prediction[0])]
    print(character)
    cv2.imshow(character,img)
    cv2.waitKey(0)