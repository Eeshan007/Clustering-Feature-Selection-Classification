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


bank_df = pd.read_csv(r"bank-additional\bank-additional-full.csv")


# In[ ]:


bank_df


# In[ ]:


bank_df['job'] = bank_df['job'].replace('unknown',bank_df['job'].mode()[0])
bank_df['marital'] = bank_df['marital'].replace('unknown',bank_df['marital'].mode()[0])
bank_df['education'] = bank_df['education'].replace('unknown',bank_df['education'].mode()[0])
bank_df['default'] = bank_df['default'].replace('unknown',bank_df['default'].mode()[0])
bank_df['housing'] = bank_df['housing'].replace('unknown',bank_df['housing'].mode()[0])
bank_df['loan'] = bank_df['loan'].replace('unknown',bank_df['loan'].mode()[0])
bank_df['pdays'] = bank_df['pdays'].replace(999,28)


# In[ ]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
bank_df["job"]=label_encoder.fit_transform(bank_df["job"])
bank_df["marital"]=label_encoder.fit_transform(bank_df["marital"])
bank_df["education"]=label_encoder.fit_transform(bank_df["education"])
bank_df["default"]=label_encoder.fit_transform(bank_df["default"])
bank_df["housing"]=label_encoder.fit_transform(bank_df["housing"])
bank_df["loan"]=label_encoder.fit_transform(bank_df["loan"])
bank_df["contact"]=label_encoder.fit_transform(bank_df["contact"])
bank_df["month"]=label_encoder.fit_transform(bank_df["month"])
bank_df["day_of_week"]=label_encoder.fit_transform(bank_df["day_of_week"])
bank_df["poutcome"]=label_encoder.fit_transform(bank_df["poutcome"])
bank_df["y"]=label_encoder.fit_transform(bank_df["y"])


# In[ ]:


normalized_df = (bank_df.iloc[:,:20] - bank_df.iloc[:,:20].mean())/bank_df.iloc[:,:20].std()


# In[ ]:


bank_df.iloc[:,:20] = normalized_df
bank_df


# In[ ]:


X=bank_df.iloc[:,:20]
y=bank_df['y']


# In[ ]:


sns.scatterplot(x=X['pdays'],y=X['day_of_week'])


# ## K Means

# In[ ]:


kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)


# In[ ]:


labels = kmeans.predict(X)


# In[ ]:


print(roc_auc_score(y,labels))
print(accuracy_score(y,labels))
print(homogeneity_score(y,labels))
print(davies_bouldin_score(X,labels))


# In[ ]:


kmeans = KMeans(n_clusters=7, random_state=12)
kmeans.fit(X)


# In[ ]:


labels = kmeans.predict(X)


# In[ ]:


print(roc_auc_score(y,labels))
print(accuracy_score(y,labels))
print(homogeneity_score(y,labels))
print(davies_bouldin_score(X,labels))


# In[ ]:


plt.scatter(x=X['age'],y=X['education'], c=labels)


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
model.add(Dense(30, input_dim=20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(7, activation='softmax'))


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


Expectation_Maximization = GaussianMixture(n_components = 7)
Expectation_Maximization.fit(X)
labels1 = Expectation_Maximization.predict(X)


# In[ ]:


print(completeness_score(y,labels1))
print(roc_auc_score(y,labels1))
print(accuracy_score(y,labels1))
print(homogeneity_score(labels,labels1))
print(davies_bouldin_score(X,labels1))
print(silhouette_score(X,labels1))


# In[ ]:


collections.Counter(labels1)


# In[ ]:


plt.scatter(x=X['education'],y=X['age'], c=labels1)


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
model.add(Dense(30, input_dim=20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(7, activation='softmax'))


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


sel.get_support()


# In[ ]:


pd.Series(sel.estimator_.feature_importances_).hist()


# In[ ]:


plt.scatter(x=X_new['age'],y=X_new['nr.employed'],c=y,cmap='rainbow')
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


sns.lineplot(x=range(1,21),y=variances)


# In[ ]:


plt.figure()
plt.figure(figsize=(8,6))
plt.xticks(fontsize=8)
plt.yticks(fontsize=10)
plt.xlabel('Principal Component - 1',fontsize=15)
plt.ylabel('Principal Component - 2',fontsize=15)
plt.title("Principal Component Analysis",fontsize=15)
targets = [0, 1]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = y == target
    plt.scatter(X_new.loc[indicesToKeep, 0]
               , X_new.loc[indicesToKeep, 1], c = color, s = 15)

plt.legend(targets,prop={'size': 15})


# In[ ]:


Principal_Component_Analysis = PCA(n_components=5)
X_new = pd.DataFrame(Principal_Component_Analysis.fit_transform(X))


# In[ ]:


X_new


# In[ ]:


kmeans = KMeans(n_clusters=5)
kmeans.fit(X_new)


# In[ ]:


labels_pca = kmeans.predict(X_new)


# In[ ]:


print(completeness_score(y,labels_pca))
print(roc_auc_score(y,labels_pca))
print(accuracy_score(labels,labels_pca))
print(homogeneity_score(labels,labels_pca))
print(davies_bouldin_score(X_new,labels_pca))
print(silhouette_score(X_new,labels_pca))


# In[ ]:


for i in range(2,16):
    Principal_Component_Analysis = PCA(n_components=i)
    X_new = pd.DataFrame(Principal_Component_Analysis.fit_transform(X))
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(X_new)
    labels_pca = kmeans.predict(X_new)
    print(davies_bouldin_score(X_new,labels_pca))
    #print(silhouette_score(X_new,labels_ica))


# In[ ]:


Principal_Component_Analysis = PCA(n_components=5)
X_new = pd.DataFrame(Principal_Component_Analysis.fit_transform(X))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.30)


# In[ ]:


model = Sequential()
model.add(Dense(40, input_dim=5, activation='relu'))
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


Independent_Component_Analysis = FastICA(n_components=5)
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
targets = [0, 1]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = y == target
    plt.scatter(X_new.loc[indicesToKeep, 0]
               , X_new.loc[indicesToKeep, 1], c = color, s = 15)

plt.legend(targets,prop={'size': 15})


# In[ ]:


kmeans = KMeans(n_clusters=5)
kmeans.fit(X_new)


# In[ ]:


labels_ica = kmeans.predict(X_new)


# In[ ]:


print(completeness_score(y,labels_ica))
print(roc_auc_score(y,labels_ica))
print(accuracy_score(labels,labels_ica))
print(homogeneity_score(labels,labels_ica))
print(davies_bouldin_score(X_new,labels_ica))
print(silhouette_score(X_new,labels_ica))


# In[ ]:


for i in range(2,10):
    Independent_Component_Analysis = FastICA(n_components=i)
    X_new = pd.DataFrame(Independent_Component_Analysis.fit_transform(X))
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(X)
    labels_ica = kmeans.predict(X)
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


# ## Randomized Projections

# In[ ]:


transformer = random_projection.GaussianRandomProjection(n_components = 11)
X_new = pd.DataFrame(transformer.fit_transform(X))


# In[ ]:


X_new


# In[ ]:


plt.figure()
plt.figure(figsize=(7,5))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Random Component - 1',fontsize=20)
plt.ylabel('Random Component - 2',fontsize=20)
plt.title("Random Component Analysis",fontsize=20)
targets = [0, 1]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = y == target
    plt.scatter(X_new.loc[indicesToKeep, 2]
               , X_new.loc[indicesToKeep, 3], c = color, s = 15)

plt.legend(targets,prop={'size': 15})


# In[ ]:


kmeans = KMeans(n_clusters=5)
kmeans.fit(X_new)


# In[ ]:


labels_rp = kmeans.predict(X_new)


# In[ ]:


print(completeness_score(y,labels_rp))
print(roc_auc_score(y,labels_rp))
print(accuracy_score(labels,labels_rp))
print(homogeneity_score(labels,labels_rp))
print(davies_bouldin_score(X_new,labels_rp))
print(silhouette_score(X_new,labels_rp))


# In[ ]:


for i in range(2,10):
    transformer = random_projection.GaussianRandomProjection(n_components = 12)
    X_new = pd.DataFrame(transformer.fit_transform(X))
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(X_new)
    labels_rp = kmeans.predict(X_new)
    print(davies_bouldin_score(X_new,labels_rp))
    #print(silhouette_score(X_new,labels_ica))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.30)


# In[ ]:


model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
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




