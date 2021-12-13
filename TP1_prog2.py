# -*- coding: utf-8 -*-
"""
-----------------------
MACHINE LEARNING - TP1
-----------------------
Léa Pitault - 5SIEC Grp C
24/11/2021
"""


from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import neighbors
import sklearn.model_selection
from sklearn.model_selection import *
import matplotlib.pyplot as plt


print('\n  ------------------------\n '
      'Mise en place du jeu de données \n'
      '   ------------------------')

mnist = fetch_openml('mnist_784', as_frame=False) 
data_train_percentage=0.8   #80% jeu de données sert pour l'apprentissage

#On prend 5000 index au hasard parmis les 70000
index=np.random.randint(70000,size=5000)
data=[mnist.data[i] for i in index]
target=[mnist.target[i] for i in index]

data_train, data_test, target_train, target_test  =  train_test_split(data, target, train_size = data_train_percentage)

#%%

print('\n   ------------------------\n '
      'Test de prediction et etude du score sur 10 voisins \n'
      '   ------------------------')

#Nombre de voisins
n_neighbors=10   

clf = neighbors.KNeighborsClassifier(n_neighbors) 
clf.fit(data_train,target_train)

print ('classe connue de la premiere image : ',target_test[0]) #

prediction=clf.predict(data_test)
print('classe predite de la premiere image: ', (prediction[0]))

score_test=clf.score(data_test,target_test)
print('Score sur echantillon test : ', score_test)

score_train=clf.score(data_train, target_train)
print('Score sur echantillon apprentissage : ',score_train)


#%%

print('\n   ------------------------\n '
      'Faire varier le nombre de voisins de 2 à 15 \n'
      '   ------------------------')
for k in range(2,16):
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(data_train,target_train)
    score = clf.score(data_test,target_test)
    print('k : ', k, 'score : ' ,score)

#%%
print('\n   ------------------------\n '
      'Faire varier le nombre de voisins de 2 à 15 \n'
      'en séparant le jeu de données en 10 sous-ensembles \n'
      '   ------------------------')
kf=KFold(n_splits=10,shuffle=True)
scores=dict()
for k in range(2,16):
    scores[k]=[]
    for indices_train,indices_test in kf.split(data):
        data_train=[data[i] for i in indices_train]
        target_train=[target[i] for i in indices_train]
        data_test =[data[i] for i in indices_test]
        target_test=[target[i] for i in indices_test]
        
        clf = neighbors.KNeighborsClassifier(k)
        clf.fit(data_train,target_train)
        
        scores[k].append(clf.score(data_test,target_test))
    
    score=np.mean(scores[k])
    print('k: ', k, '; score :', score)
    
#%%
print('\n   ------------------------\n '
      'Faire varier le pourcentage des echantillons \n'
      '   ------------------------')

n_neighbors=10
scores=[]  
percentages=[] 
  
for data_train_percentage in range(5, 95, 5): 

    index=np.random.randint(70000,size=5000)
    data=[mnist.data[i] for i in index]
    target=[mnist.target[i] for i in index]

    data_train, data_test, target_train, target_test  =  train_test_split(data, target, train_size = data_train_percentage/100)


    clf = neighbors.KNeighborsClassifier(n_neighbors) 
    clf.fit(data_train,target_train)
    
    print('\n Pourcentage de training: ', data_train_percentage)
    percentages.append(data_train_percentage)
    
    print ('classe connue de la premiere image : ',target_test[0]) #

    prediction=clf.predict(data_test)
    print('classe predite de la premiere image: ', (prediction[0]))
    
    score_test=clf.score(data_test,target_test)
    print('Score sur echantillon test : ', score_test)
    scores.append(score_test)
    
    score_train=clf.score(data_train, target_train)
    print('Score sur echantillon apprentissage : ',score_train)
    
    
    #tracer la courbe score en fonction porucentage echantillon test
plt.plot(percentages, scores)
plt.title('Score en fonction du pourcentage de valeur pour l\'entrainement')
plt.xlabel('Pourcentage de valeur d\'entrainement')
plt.ylabel('Score')
plt.show()
    
    #%%
print('\n   ------------------------\n '
      'Faire varier la taille des echantillons \n'
      '   ------------------------')

n_neighbors=10   
data_train_percentage=0.8
scores=[]  
nb_images=[] 

for size_samples in range(5000, 55000, 5000): 

    index=np.random.randint(70000,size=size_samples)
    data=[mnist.data[i] for i in index]
    target=[mnist.target[i] for i in index]

    data_train, data_test, target_train, target_test  =  train_test_split(data, target, train_size = data_train_percentage)


    clf = neighbors.KNeighborsClassifier(n_neighbors) 
    clf.fit(data_train,target_train)
    
    print('\n Nb de d\'images total pris sur les 70000 : ', size_samples)
    nb_images.append(size_samples)
    #print ('classe connue de la premiere image : ',target_test[0]) #

    prediction=clf.predict(data_test)
    #print('classe predite de la premiere image: ', (prediction[0]))
    
    score_test=clf.score(data_test,target_test)
    print('Score sur echantillon test : ', score_test)
    scores.append(score_test)
    #score_train=clf.score(data_train, target_train)
    #print('Score sur echantillon apprentissage : ',score_train)
    
#tracer la courbe score en fonction du nb d'image prises dans la base de données
plt.plot(nb_images, scores)
plt.title('Score en fonction du nombre d\'image')
plt.xlabel('Nombre d\'images prises parmi les 70 000')
plt.ylabel('Score')
plt.show()