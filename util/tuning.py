#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def mean_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level=0).mean()
df = pd.read_hdf('/Users/bailujia/Downloads/train.h5', index_col='id')


# In[9]:


folds = pd.read_csv('/Users/bailujia/Downloads/folds.csv')
validset = folds.loc[folds['fold'] != 0]
validset.index = range(validset.shape[0])
df = df.reindex(validset['id'])
df = mean_df(df)


# In[10]:


import numpy as np
def get_target():
    target = np.zeros((validset.shape[0],1103))
    index = validset['attribute_ids'].str.split()
    for i in range(validset.shape[0]):
        label = np.array([int(cls) for cls in index[i]])
        target[i,label] = 1
    return target


# In[12]:


target = get_target()


# In[13]:


#class 0
target[:,0]


# In[14]:


X= np.array(df)


# In[15]:


print(X.shape)


# In[16]:


from collections import defaultdict, Counter
import random
import pandas as pd
import tqdm


# In[17]:


def make_folds(df, n_folds: int) -> pd.DataFrame:
    cls_counts = Counter(cls for classes in df['attribute_ids'].str.split()
                         for cls in classes)
    fold_cls_counts = defaultdict(int)
    folds = [-1] * len(df)
    for item in tqdm.tqdm(df.sample(frac=1, random_state=42).itertuples(),
                          total=len(df)):
        cls = min(item.attribute_ids.split(), key=lambda cls: cls_counts[cls])
        fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
        min_count = min([count for _, count in fold_counts])
        random.seed(item.Index)
        fold = random.choice([f for f, count in fold_counts
                              if count == min_count])
        folds[item.Index] = fold
        for cls in item.attribute_ids.split():
            fold_cls_counts[fold, cls] += 1
    df['fold'] = folds
    return df


# In[18]:


kf = make_folds(validset,5)


# In[40]:


X = np.array(df)


# In[47]:


np.array(ind)


# In[46]:


ind = []
for k in range(1103):
    if np.sum(target[:,k] ==0)*np.sum(target[:,k] ==1)>0:
        if np.max(X[target[:,k] ==0 ,k]) < np.min(X[target[:,k] ==1 ,k]):
            if np.min(X[target[:,k] ==1 ,k])>0.9:
                if X[target[:,k] ==1,k].shape[0] > 5:
                    ind.append(k)
                    print(k,np.max(X[target[:,k] ==0 ,k]), np.min(X[target[:,k] ==1 ,k]))


# In[34]:


np.array(ind)


# In[45]:


ind = []
for k in range(1103):
    if np.sum(target[:,k] ==0)*np.sum(target[:,k] ==1)>0:
        if np.max(X[target[:,k] ==0 ,k]) < np.min(X[target[:,k] ==1 ,k]):
            if 0.9>= np.min(X[target[:,k] ==1 ,k])>0.8:
                if X[target[:,k] ==1,k].shape[0] > 5:
                    ind.append(k)
                    print(k,np.max(X[target[:,k] ==0 ,k]), np.min(X[target[:,k] ==1 ,k]))


# In[44]:


np.array(ind)


# In[43]:


ind = []
for k in range(1103):
    if np.sum(target[:,k] ==0)*np.sum(target[:,k] ==1)>0:
        if np.max(X[target[:,k] ==0 ,k]) < np.min(X[target[:,k] ==1 ,k]):
            if 0.8>= np.min(X[target[:,k] ==1 ,k])>0.7:
                if X[target[:,k] ==1,k].shape[0] > 5:
                    ind.append(k)
                    print(k,np.max(X[target[:,k] ==0 ,k]), np.min(X[target[:,k] ==1 ,k]))


# In[ ]:


np.array(ind)


# In[50]:


np.linspace(0.2,0.7,6)


# In[55]:


for th in np.linspace(0.2,0.7,6):
    print(th)
    ind = []
    for k in range(1103):
        if np.sum(target[:,k] ==0)*np.sum(target[:,k] ==1)>0:
            if np.max(X[target[:,k] ==0 ,k]) < np.min(X[target[:,k] ==1 ,k]):
                if (0.1+th)>= np.min(X[target[:,k] ==1 ,k])> th:
                    if X[target[:,k] ==1,k].shape[0] > 5:
                        ind.append(k)
                        m1 = np.max(X[target[:,k] ==0 ,k])
                        m2 =  np.min(X[target[:,k] ==1 ,k])
                        print(k,m1,m2,np.mean([m1,m2]))
    print("index:",np.array(ind))


# In[81]:


ind


# In[80]:


ind = []
for k in range(1103):
    if np.sum(target[:,k] ==0)*np.sum(target[:,k] ==1)>0:
        if (np.max(X[target[:,k] ==0 ,k]) - np.min(X[target[:,k] ==1 ,k]))< -0.005 :
             if X[target[:,k] ==1,k].shape[0] > 3:
                ind.append([k,np.round(0.1*m1+0.9*m2,2)])
                m1 = np.max(X[target[:,k] ==0 ,k])
                m2 =  np.min(X[target[:,k] ==1 ,k])
                #print(k,m1,m2,0.1*m1+0.9*m2)


# In[85]:


ind = []
for k in range(1103):
    if np.sum(target[:,k] ==0)*np.sum(target[:,k] ==1)>0:
        if (np.max(X[target[:,k] ==0 ,k]) - np.min(X[target[:,k] ==1 ,k]))> -0.005 :
             if X[target[:,k] ==1,k].shape[0] > 3:
                ind.append([k,np.round(0.1*m1+0.9*m2,2)])
                m1 = np.max(X[target[:,k] ==0 ,k])
                m2 =  np.min(X[target[:,k] ==1 ,k])
                print(k,m1,np.quantile(X[target[:,k] ==0 ,k],0.9999),m2)


# In[108]:


ind = []
for k in range(1103):
    if np.sum(target[:,k] ==0)*np.sum(target[:,k] ==1)>0:
        if (np.max(X[target[:,k] ==0 ,k]) - np.min(X[target[:,k] ==1 ,k]))> -0.005 and        (np.quantile(X[target[:,k] ==0 ,k],0.99) - np.min(X[target[:,k] ==1 ,k]))< 0 :
             if X[target[:,k] ==1,k].shape[0] > 3:
                m1 =  np.quantile(X[target[:,k] ==0 ,k],0.99)
                m2 =  np.min(X[target[:,k] ==1 ,k])
                ind.append([k,round(np.quantile(X[target[:,k] ==1 ,k],0.3),2)])
                #print(k,m1,m2,0.1*m1+0.9*m2)


# In[109]:


ind


# In[122]:


k = 64
import matplotlib.pyplot as plt
mu1 = np.mean(X[target[:,k] ==0 ,k],)
mu2 = np.mean(X[target[:,k] ==1 ,k])
sigma1 = np.std(X[target[:,k] ==0 ,k])
sigma2 = np.std(X[target[:,k] ==1 ,k])

plt.subplot(1,2,1)
plt.boxplot(x = X[target[:,k] ==0 ,k],patch_artist = True,)
plt.ylim(0,1)
plt.subplot(1,2,2)
plt.boxplot(x = X[target[:,k] ==1 ,k],patch_artist = True,)
plt.ylim(0,1)
print((mu1 * sigma2 +  mu2 * sigma1)/(sigma1 + sigma2))
print((mu1 +  mu2)/2)


