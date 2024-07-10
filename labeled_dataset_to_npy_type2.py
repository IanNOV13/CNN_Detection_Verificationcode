import os,cv2,glob
import numpy as np
from sklearn.model_selection import train_test_split

label_folder = './labeled_dataset/type2/'
save_folder = './numpy_data/a/'

images = []
labels = []
characters = 'abcdefghijklmnopqrstuvwxyz'
num_classes = len(characters)
dict_labels = {char: idx for idx, char in enumerate(characters)}

char_counter = [
    [0]*26 for _ in range(4)
    ]

size = (120,100)
for paths in os.listdir(label_folder):
    print(f'開始掃描{os.path.join(label_folder,paths)}')
    for dirs in os.listdir(os.path.join(label_folder,paths)):
        path = os.path.join(label_folder,paths,dirs)
        for file in os.listdir(path):
            img = cv2.imread(f'{path}/{file}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
            new_label = []
            for i,c in enumerate(dirs):
                char_counter[i][dict_labels[c]] += 1
                new_label.append(dict_labels[c])
            labels.append(new_label)
    print(f'{os.path.join(label_folder,paths)}完成掃描!')

print(f"總共{len(images)}張照片，{len(labels)}個標籤!")

train_feature,test_feature,train_label,test_label = train_test_split(images,labels,test_size=0.1,random_state=42)

train_feature = np.array(train_feature)
test_feature = np.array(test_feature)
train_label = np.array(train_label)
test_label = np.array(test_label)
char_counter = np.array(char_counter)
char_frequencies = char_counter / np.sum(char_counter)
weights = 1.0/char_frequencies

print(f"訓練集資料共{len(train_feature)}筆，測試集資料共{len(test_feature)}筆!")
print("".join("*" for _ in range(10)))
print(f"訓練集資料維度\n照片:{train_feature.shape}\n標籤:{train_label.shape}")
print("".join("*" for _ in range(10)))
print(f"測試集資料維度\n照片:{test_feature.shape}\n標籤:{test_label.shape}")
print("".join("*" for _ in range(10)))
for i,data in enumerate(weights):
    print("第%s權重資料:\n%s"%(i+1,data))
print("".join("*" for _ in range(10)))
np.save(save_folder+'train_feature.npy',train_feature)
np.save(save_folder+'train_label.npy',train_label)
np.save(save_folder+'test_feature.npy',test_feature)
np.save(save_folder+'test_label.npy',test_label)
np.save(save_folder+'class_weights',weights)

print('npy檔存檔完成!')