import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.decomposition import PCA

print('Read data...')
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[:1000,1:].values.astype(np.float)
labels = labeled_images.iloc[:1000,:1].values.astype(np.int16)
# images[images < 100] = 0
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

test_data = pd.read_csv('../input/test.csv').iloc[:,:].values.astype(np.float)
# test_data[test_data < 100] = 0

print('PCA...')
pca = PCA(n_components=0.8, whiten=True)
train_images = pca.fit_transform(train_images)
val_images = pca.transform(val_images)
test_data = pca.transform(test_data)

params={
    'booster':'gbtree',
    # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器
    'objective': 'multi:softmax',
    'num_class':10, # 类数，与 multisoftmax 并用
    # 'gamma':,  # 在树的叶子节点下一个分区的最小损失，越大算法
    # 模型越保守 。[0:]
    'max_depth':9, # 构建树的深度 [1:]
    #'lambda':450,  # L2 正则项权重
    'subsample':0.7, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree':0.8, # 构建树树时的采样比率 (0:1]
    #'min_child_weight':12, # 节点的最少特征数
    'silent':1,
    'eta': 0.1, # 如同学习率
    'seed':4321,

    'nthread':2,# cpu 线程数,根据自己U的个数适当调整
}

plst = params

xgtrain = xgb.DMatrix(train_images, label=train_labels)
xgval = xgb.DMatrix(val_images, label=val_labels)
xgtest = xgb.DMatrix(test_data)
watchlist = [(xgtrain, 'train'),(xgval, 'val')]

print('Train SVM...')
num_rounds=200
model = xgb.train(plst, xgtrain, num_rounds, watchlist)
print('Predicting...')
results = model.predict(xgtest).astype(np.int16)

print('Saving...')
pd.DataFrame({"ImageId": list(range(1, len(results) + 1)), "Label": results}).to_csv("result.csv", index=False, header=True)

