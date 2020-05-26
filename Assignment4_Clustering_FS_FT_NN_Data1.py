#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from statistics import mean


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.utils.np_utils import to_categorical


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import completeness_score, homogeneity_score, davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn import random_projection
from sklearn.decomposition import FastICA, PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# In[ ]:


project_df = pd.read_csv("sgemm_product_dataset\sgemm_product.csv")


# In[ ]:


project_df['Run_Avg'] = project_df.iloc[:,14:18].mean(axis=1)


# In[ ]:


project_df=project_df.drop(columns=['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'])


# In[ ]:


project_df=project_df.dropna()


# In[ ]:


project_df['Run_Avg'].median()
project_df['Run_Avg'] = np.where(project_df['Run_Avg'] >= project_df['Run_Avg'].median(), 1, 0)
#Converted all the values above median to 1 and below median to zero


# In[ ]:


normalized_df = (project_df.iloc[:,:14] - project_df.iloc[:,:14].mean())/project_df.iloc[:,:14].std()


# In[ ]:


project_df.iloc[:,:14] = normalized_df


# In[ ]:


project_df=project_df.sample(frac=0.3, replace=False, random_state=1)
project_df


# In[ ]:


X=project_df.iloc[:,:14]
y=project_df['Run_Avg']


# ## K Means

# In[ ]:


kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)
labels = kmeans.predict(X)


# In[ ]:


print(roc_auc_score(y,labels))
print(accuracy_score(y,labels))
print(homogeneity_score(y,labels))
print(davies_bouldin_score(X,labels))
print(silhouette_score(X,labels))


# In[ ]:


kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X)
labels = kmeans.predict(X)


# In[ ]:


print(roc_auc_score(y,labels))
print(accuracy_score(y,labels))
print(homogeneity_score(y,labels))
print(davies_bouldin_score(X,labels))
print(silhouette_score(X,labels))


# In[ ]:


plt.scatter(x=X['MWG'],y=X['VWN'], c=labels)


# In[ ]:


sns.countplot(labels)


# In[ ]:


scores = [KMeans(n_clusters=i+2).fit(X).inertia_ 
          for i in range(10)]


# In[ ]:


sns.lineplot(np.arange(2, 12), scores)


# In[ ]:


y_new = labels


# In[ ]:


collections.Counter(y_new)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y_new, stratify=y_new, test_size = 0.30)


# In[ ]:


y_train = to_categorical(y_train)
y_train


# In[ ]:


model = Sequential()
model.add(Dense(30, input_dim=14, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(5, activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mean_squared_error','accuracy'])
history = model.fit(X_train, y_train, epochs=100, verbose=0)


# In[ ]:


_,_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


# In[ ]:


predictions = model.predict_classes(X_test)


# In[ ]:


y_test = to_categorical(y_test)
y_test


# In[ ]:


_,_, accuracy3 = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy3*100))


# In[ ]:


plt.plot(history.history['mean_squared_error'])
plt.show()
plt.plot(history.history['accuracy'],color='red')
plt.show()


# ## Expectation Maximization

# In[ ]:


Expectation_Maximization = GaussianMixture(n_components = 5)
Expectation_Maximization.fit(X)
labels1 = Expectation_Maximization.predict(X)


# In[ ]:


print(completeness_score(y,labels1))
print(roc_auc_score(y,labels1))
print(accuracy_score(y,labels1))
print(homogeneity_score(labels,labels1))
print(davies_bouldin_score(X,labels1))


# In[ ]:


collections.Counter(labels1)


# In[ ]:


plt.scatter(x=X['MWG'],y=X['VWN'], c=labels1)


# In[ ]:


sns.countplot(labels1)


# In[ ]:


y_new = labels1


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y_new, stratify=y_new, test_size = 0.30)


# In[ ]:


y_train = to_categorical(y_train)
y_train


# In[ ]:


model = Sequential()
model.add(Dense(30, input_dim=14, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(5, activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mean_squared_error','accuracy'])
history = model.fit(X_train, y_train, epochs=100, verbose=0)


# In[ ]:


_,_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


# In[ ]:


predictions = model.predict_classes(X_test)


# In[ ]:


y_test = to_categorical(y_test)
y_test


# In[ ]:


_,_, accuracy3 = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy3*100))


# In[ ]:


plt.plot(history.history['mean_squared_error'])
plt.show()
plt.plot(history.history['accuracy'],color='red')
plt.show()


# ## Decision Tree

# In[ ]:


sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X, y)
sel.get_support()
selected_columns= X.columns[(sel.get_support())]
X_new = X[selected_columns]
length = len(selected_columns)


# In[ ]:


pd.Series(sel.estimator_.feature_importances_.ravel()).hist()


# In[ ]:


plt.scatter(x=X_new['MWG'],y=X_new['NWG'],c=y,cmap='rainbow')
plt.legend(y,prop={'size': 5})


# In[ ]:


kmeans = KMeans(n_clusters=5)
kmeans.fit(X_new)


# In[ ]:


labels_random_forest = kmeans.predict(X_new)


# In[ ]:


print(completeness_score(y,labels_random_forest))
print(roc_auc_score(y,labels_random_forest))
print(accuracy_score(labels,labels_random_forest))
print(homogeneity_score(labels,labels_random_forest))
print(davies_bouldin_score(X_new,labels_random_forest))
print(silhouette_score(X_new,labels_random_forest))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.30)


# In[ ]:


model = Sequential()
model.add(Dense(40, input_dim=length, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)


# In[ ]:


_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


# In[ ]:


predictions = model.predict_classes(X_test)


# In[ ]:


_, accuracy2 = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy2*100))


# ## PCA

# In[ ]:


Principal_Component_Analysis = PCA()
X_new = pd.DataFrame(Principal_Component_Analysis.fit_transform(X))


# In[ ]:


variances=Principal_Component_Analysis.explained_variance_ratio_


# In[ ]:


Principal_Component_Analysis.explained_variance_ratio_


# In[ ]:


sns.lineplot(x=range(1,15),y=variances)


# In[ ]:


Principal_Component_Analysis = PCA(n_components=12)
X_new = pd.DataFrame(Principal_Component_Analysis.fit_transform(X))


# In[ ]:


X_new


# In[ ]:


plt.figure()
plt.figure(figsize=(8,6))
plt.xticks(fontsize=8)
plt.yticks(fontsize=10)
plt.xlabel('Principal Component - 1',fontsize=15)
plt.ylabel('Principal Component - 2',fontsize=15)
plt.title("Principal Component Analysis",fontsize=15)
plt.scatter(x=X_new[0],y=X_new[1], c=y)


# In[ ]:


kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_new)


# In[ ]:


labels_pca = kmeans.predict(X_new)


# In[ ]:


print(completeness_score(y,labels_pca))
print(roc_auc_score(y,labels_pca))
print(accuracy_score(labels,labels_pca))
print(homogeneity_score(labels,labels_pca))
print(davies_bouldin_score(X_new,labels_pca))


# In[ ]:


for i in range(2,14):
    Principal_Component_Analysis = PCA(n_components=i)
    X_new = pd.DataFrame(Principal_Component_Analysis.fit_transform(X))
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(X_new)
    labels_pca = kmeans.predict(X_new)
    print(davies_bouldin_score(X_new,labels_pca))
    #print(silhouette_score(X_new,labels_ica))


# In[ ]:


Principal_Component_Analysis = PCA(n_components=12)
X_new = pd.DataFrame(Principal_Component_Analysis.fit_transform(X))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.30)


# In[ ]:


model = Sequential()
model.add(Dense(40, input_dim=12, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)


# In[ ]:


_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


# In[ ]:


predictions = model.predict_classes(X_test)


# In[ ]:


_, accuracy2 = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy2*100))


# ## ICA

# In[ ]:


Independent_Component_Analysis = FastICA(n_components=6)
X_new = pd.DataFrame(Independent_Component_Analysis.fit_transform(X))


# In[ ]:


X_new


# In[ ]:


plt.figure()
plt.figure(figsize=(8,6))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Independent Component - 1',fontsize=20)
plt.ylabel('Independent Component - 2',fontsize=20)
plt.title("Independent Component Analysis",fontsize=20)
plt.scatter(X_new[0], X_new[1], c = y)


# In[ ]:


kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_new)


# In[ ]:


labels_ica = kmeans.predict(X_new)


# In[ ]:


print(completeness_score(y,labels_ica))
print(roc_auc_score(y,labels_ica))
print(accuracy_score(labels,labels_ica))
print(homogeneity_score(labels,labels_ica))
print(davies_bouldin_score(X_new,labels_ica))


# In[ ]:


for i in range(2,14):
    Independent_Component_Analysis = FastICA(n_components=i)
    X_new = pd.DataFrame(Independent_Component_Analysis.fit_transform(X))
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(X_new)
    labels_ica = kmeans.predict(X_new)
    print(davies_bouldin_score(X_new,labels_ica))
    #print(silhouette_score(X_new,labels_ica))


# In[ ]:


Independent_Component_Analysis = FastICA(n_components=5)
X_new = pd.DataFrame(Independent_Component_Analysis.fit_transform(X))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.30)


# In[ ]:


model = Sequential()
model.add(Dense(30, input_dim=5, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)


# In[ ]:


_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


# In[ ]:


predictions = model.predict_classes(X_test)


# In[ ]:


_, accuracy2 = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy2*100))


# ## Randomized Projections

# In[ ]:


transformer = random_projection.GaussianRandomProjection(n_components = 9)
X_new = pd.DataFrame(transformer.fit_transform(X))


# In[ ]:


X_new


# In[ ]:


plt.figure()
plt.figure(figsize=(8,6))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Random Component - 1',fontsize=20)
plt.ylabel('Random Component - 2',fontsize=20)
plt.title("Random Component Analysis",fontsize=20)
plt.scatter(X_new[0], X_new[1], c = y)


# In[ ]:


kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X_new)


# In[ ]:


labels_rp = kmeans.predict(X_new)


# In[ ]:


print(completeness_score(y,labels_rp))
print(roc_auc_score(y,labels_rp))
print(accuracy_score(labels,labels_rp))
print(homogeneity_score(labels,labels_rp))
print(davies_bouldin_score(X_new,labels_rp))


# In[ ]:


for i in range(2,14):
    transformer = random_projection.GaussianRandomProjection(n_components = i)
    X_new = pd.DataFrame(transformer.fit_transform(X))
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(X_new)
    labels_rp = kmeans.predict(X_new)
    print(davies_bouldin_score(X_new,labels_rp))
    #print(silhouette_score(X_new,labels_ica))


# In[ ]:


transformer = random_projection.GaussianRandomProjection(n_components = 11)
X_new = pd.DataFrame(transformer.fit_transform(X))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.30)


# In[ ]:


model = Sequential()
model.add(Dense(30, input_dim=11, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)


# In[ ]:


_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


# In[ ]:


predictions = model.predict_classes(X_test)


# In[ ]:


_, accuracy2 = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy2*100))


# In[ ]:




