# -*- coding: utf-8 -*-
"""
-----------------------
MACHINE LEARNING - TP2
-----------------------
Léa Pitault - 5SIEC Grp C
Date début : 06/12/2021
Date rendu : 15/12/2021
Enseignant : Adrien Dorise
"""


from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import neighbors
import sklearn.model_selection
from sklearn.model_selection import *
import matplotlib.pyplot as plt
from sklearn.neural_network import *
from sklearn.metrics import *
import time
from random import randint


print('\n  ------------------------\n '
      'Mise en place du jeu de données \n'
      '   ------------------------')

mnist = fetch_openml('mnist_784', as_frame=False) 
data_train_percentage=0.7   #70% jeu de données sert pour l'apprentissage, soit 49000 données

data=mnist.data
target=mnist.target

data_train, data_test, target_train, target_test  =  train_test_split(data, target, train_size = data_train_percentage)

#%%
#Modèle simple avec 1 couche de 50 neurones
#affichage de la classe de la première image et de la classe prédite
clf = MLPClassifier(hidden_layer_sizes=(50))
clf.fit(data_train, target_train)

score_test=clf.score(data_test,target_test)
print('Score sur echantillon test : ', score_test)

print ('classe connue de la premiere image : ',target_test[0]) #

prediction=clf.predict(data_test)
print('classe predite de la premiere image: ', (prediction[0]))

#precision avec package
precision=precision_score(target_test,prediction,average='micro')
print('precision avec package: ', (precision))

#%%
#Varier le nombre neurones entre 2 et 100 pour une seule couche.
precision=[]
recall_scores=[]
times=[]
error=[]
neurones=[]
for i in range(2,101) :
    print('i : ',i)
    neurones.append(i)
    clf = MLPClassifier(hidden_layer_sizes=(i),max_iter=10)
    start = time.time()
    clf.fit(data_train, target_train)
    stop=time.time()
    training_time=stop-start
    times.append(round(training_time,2))
    prediction=clf.predict(data_test)
    precision.append(precision_score(target_test,prediction,average='micro'))
    recall_scores.append(recall_score(target_test, prediction, average='micro'))
    error.append(zero_one_loss(target_test, prediction))
    
plt.plot(neurones,times)
plt.grid()
plt.xlabel('Nb de neurones')
plt.ylabel('Temps')
plt.title('Temps d\'apprentissage en fonction du nombre de neurone')
plt.show()

    
plt.plot(neurones,precision)
plt.grid()
plt.xlabel('Nb de neurone')
plt.ylabel('Precision')
plt.title('Precision en fonction du nombre de neurone')
plt.show()

plt.plot(neurones,recall_scores)
plt.grid()
plt.xlabel('Nb de neurone')
plt.ylabel('Score de rappel')
plt.title('Score de rappel en fonction du nombre de neurone')
plt.show()

plt.plot(neurones,error)
plt.grid()
plt.xlabel('Nb de neurone')
plt.ylabel('Erreur zero_one_loss')
plt.title('Erreur zero_one_loss en fonction du nombre de neurone')
plt.show()


    #essayer de faire 1couche, 20 couches, 40 couches, etc... jusqu'à 100 mais avec un seul neurone
    
    
#%%
#Construire des réseaux de 1 à 10 couches cachées 
#avec des tailles de couches entre 10 et 300 neurones aléatoire.
times2={}
precision2={}
recall_scores2={}
error2={}

neurones = []

#Nombre aléatoire de neurones entre 10 et 300
neurones.append(randint(10,301))

#Nombre de couche entre 1 et 10
for i in range(1,11):
    print(i)
    clf = MLPClassifier(hidden_layer_sizes=neurones,max_iter=10)
    start = time.time()
    clf.fit(data_train, target_train)
    stop = time.time()
    training_time = stop-start
    times2[i] = round(training_time,2)
    prediction = clf.predict(data_test)
    precision2[i] = precision_score(target_test,prediction,average='micro')
    recall_scores2[i] = (recall_score(target_test, prediction, average='micro'))
    error2[i] = (zero_one_loss(target_test, prediction))
    
    neurones.append(randint(10,300))

plt.plot(times2.keys(),times2.values())
plt.grid()
plt.xlabel('Nb de couche')
plt.ylabel('Temps (s)')
plt.title('Temps d\'apprentissage en fonction du nombre de couche')
plt.show()

    
plt.plot(precision2.keys(),precision2.values())
plt.grid()
plt.xlabel('Nb de couche')
plt.ylabel('Precision')
plt.title('Precision en fonction du nombre de couche')
plt.show()

plt.plot(recall_scores2.keys(),recall_scores2.values())
plt.grid()
plt.xlabel('Nb de couche')
plt.ylabel('Score de rappel')
plt.title('Score de rappel en fonction du nombre de couche')
plt.show()

plt.plot(error2.keys(),error2.values())
plt.grid()
plt.xlabel('Nb de couche')
plt.ylabel('Erreur zero_one_loss')
plt.title('Erreur zero_one_loss en fonction du nombre de couche')
plt.show()


plt.plot(times2.keys(),neurones[:10])
plt.grid()
plt.xlabel('Nb de couche')
plt.ylabel('Nb de neurones')
plt.title('Nb de neurones selon la couche')
plt.show()

print(neurones)




#%%
#Construire un modèle pour tester les différents solveurs

times3={}
precision3={}
recall_scores3={}
error3={}


for i in ["lbfgs","sgd","adam"]:
    print(i)
    #on fait le choix de 5 couches et du nombre de neurones en reprenant les résultats précédents
    clf = MLPClassifier(solver=i, hidden_layer_sizes=(74,297,27,191,23),max_iter=10)
    start = time.time()
    clf.fit(data_train, target_train)
    stop = time.time()
    training_time = stop-start
    times3[i] = round(training_time,2)
    prediction = clf.predict(data_test)
    
    precision3[i] = precision_score(target_test,prediction,average='micro')
    recall_scores3[i] = (recall_score(target_test, prediction, average='micro'))
    error3[i] = (zero_one_loss(target_test, prediction))
    
    neurones.append(randint(10,300))

    print('\n Solveur : ', i)
    print('Temps d\'apprentissage : ', times3[i])
    print('Précision : ', precision3[i])
    print('Recall score : ', recall_scores3[i])
    print('Erreur zero_one_loss : ', error3[i])



#%%
#Modèle pour étudier les fonctions d'activation

times4={}
precision4={}
recall_scores4={}
error4={}
index=[]

for i in ["identity","logistic","tanh", "relu"]:
    print(i)
    index.append(i)

    clf = MLPClassifier(hidden_layer_sizes=(74,297,27,191,23),max_iter=10, activation=i)
    start = time.time()
    clf.fit(data_train, target_train)
    stop = time.time()
    training_time = stop-start
    times4[i] = round(training_time,2)
    prediction = clf.predict(data_test)
    
    precision4[i] = precision_score(target_test,prediction,average='micro')
    recall_scores4[i] = (recall_score(target_test, prediction, average='micro'))
    error4[i] = (zero_one_loss(target_test, prediction))
    
plt.plot(index,times4.values())
plt.grid()
plt.xlabel('Fonction d\'activation')
plt.ylabel('Temps (s)')
plt.title('Temps d\'apprentissage en fonction de la fonction d\'activation')
plt.show()

plt.plot(index,precision4.values())
plt.grid()
plt.xlabel('Fonction d\'activation')
plt.ylabel('Precision')
plt.title('Precision en fonction de la fonction d\'activation')
plt.show()

plt.plot(index,recall_scores4.values())
plt.grid()
plt.xlabel('Fonction d\'activation')
plt.ylabel('Recall score')
plt.title('Recall score en fonction de la fonction d\'activation')
plt.show()

plt.plot(index,error4.values())
plt.grid()
plt.xlabel('Fonction d\'activation')
plt.ylabel('Erreur')
plt.title('Erreur zero_one_loss en fonction de la fonction d\'activation')
plt.show()

#%%
#Modèle pour étudier le paramètre alpha (valeur de régularisation)

times6={}
precision6={}
recall_scores6={}
error6={}
index=[]

alpha=[100,500,1000,5000,7500,10000,25000,50000,75000,100000]

for i in alpha:
    print(1/i)
    index.append(1/i)

    clf = MLPClassifier(hidden_layer_sizes=(74,297,27,191,23),max_iter=10, alpha=(1/i))
    start = time.time()
    clf.fit(data_train, target_train)
    stop = time.time()
    training_time = stop-start
    times6[i] = round(training_time,2)
    prediction = clf.predict(data_test)
    
    precision6[i] = precision_score(target_test,prediction,average='micro')
    recall_scores6[i] = (recall_score(target_test, prediction, average='micro'))
    error6[i] = (zero_one_loss(target_test, prediction))
    
plt.plot(index,times6.values())
plt.grid()
plt.xlabel('Valeur de régularisation')
plt.ylabel('Temps (s)')
plt.title('Temps d\'apprentissage en fonction de la valeur de régularisation')
plt.show()

    
plt.plot(index,precision6.values())
plt.grid()
plt.xlabel('Valeur de régularisation')
plt.ylabel('Precision')
plt.title('Precision en fonction de la valeur de régularisation')
plt.show()

plt.plot(index,recall_scores6.values())
plt.grid()
plt.xlabel('Valeur de régularisation')
plt.ylabel('Recall score')
plt.title('Recall score en fonction de la valeur de régularisation')
plt.show()

plt.plot(index,error6.values())
plt.grid()
plt.xlabel('Valeur de régularisation')
plt.ylabel('Erreur')
plt.title('Erreur zero_one_loss en fonction de la valeur de régularisation')
plt.show()


#%%
#Construire des réseaux de 1 à 100 couches cachées avec un pas de 20
#avec des tailles de couches de 1 neurone
times7={}
precision7={}
recall_scores7={}
error7={}

neurones = []

neurones.append(1)

#Nombre de couche entre 1 et 100
for i in range(1,101):
    print(i)
    print(neurones)

    clf = MLPClassifier(hidden_layer_sizes=neurones,max_iter=10)
    start = time.time()
    clf.fit(data_train, target_train)
    stop = time.time()
    training_time = stop-start
    times7[i] = round(training_time,2)
    prediction = clf.predict(data_test)
    precision7[i] = precision_score(target_test,prediction,average='micro')
    recall_scores7[i] = (recall_score(target_test, prediction, average='micro'))
    error7[i] = (zero_one_loss(target_test, prediction))
    
    neurones.append(1)
   
#affichage des valeurs pour 1,20,40,60,80,100 couches
for i in range(1,101,20):
    if i==1:
        print('\n Valeurs pour', i,' couches à 1 neurone')
        print('Temps d\'apprentissage (s): ', times7[i])
        print('Précision : ', precision7[i])
        print('Recall score : ', recall_scores7[i])
        print('Erreur zero_one_loss : ', error7[i])
    else:
        print('\n Valeurs pour', i-1,' couches à 1 neurone')
        print('Temps d\'apprentissage (s): ', times7[i-1])
        print('Précision : ', precision7[i-1])
        print('Recall score : ', recall_scores7[i-1])
        print('Erreur zero_one_loss : ', error7[i-1])

print('\n Valeurs pour', 100,' couches à 1 neurone')
print('Temps d\'apprentissage (s): ', times7[100])
print('Précision : ', precision7[100])
print('Recall score : ', recall_scores7[100])
print('Erreur zero_one_loss : ', error7[100])

    
    
    
    