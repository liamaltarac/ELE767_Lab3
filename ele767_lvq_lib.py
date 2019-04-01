#######################################################
##  Fichier : ELE767_lvq_lib.py
##  Auteurs : G. Cavero , L. Frija, S. Mohammed
##  Date de creation : 20 Mars 2019          
##  Description : Ce fichier contient toutes les
##                fonctions qui servent a entrainner  
##                un RN LVQ comme on a vu en classe
##
##
#######################################################

from couche import Couche
import logging, sys
import numpy as np
import random
import re
import os
from scipy.spatial import distance

from time import time

debug = True

class LVQ(object):
    #Fonction pour l'initilisation des paramètres du LVQ
    def __init__(self, numEntrees = None, 
                eta = 0.1, sortiePotentielle = None, 
                epoche = 1, etaAdaptif = False, perf_VC = 0.75, 
                VCin = None, VCout = None, fichier_lvq = None, k = 10, fichierReps = None):   

        self.numEntrees = numEntrees
        print("# entrees ", self.numEntrees)


        self.eta = eta
        self.etaInit = eta
        self.sortiesPotentielle = sortiePotentielle
        self.epoche = epoche
        self.performance  = np.array([])
        self.performanceVC = np.array([])

        self.etaAdaptif = etaAdaptif   

        self.couches = [] #Creer une liste de toute les couches du MLP, commencant par les couches cachees,
                          #et terminant par la couche de sortie.

        self.perf_VC = 0
        self.perf_ENT = 0

        self.VCin = VCin
        self.VCout = VCout
        
        self.totalNumEpoche = 0
        print("num Entrees = ", numEntrees)

        self.k = k

        if fichier_lvq is not None:
            seuilsArray = []
            with open(fichier_lvq,'r') as f:
                data = f.read().replace(" ", "").lower()  #On enleve tout les espaces pour eviter d'avoir une erreur

            for line in data.split("\n"):
                if len(line) > 1:

                    key = line.split("=")[0]

                    value = line.split("=")[1]
                    if key == "k":
                        self.k = eval(value)
                        print("K = ", k)
                    elif key == "eta":
                        self.eta = eval(value)
                        self.etaInit = self.eta
                        print("eta = ", eta)

                    elif key == "nb_entrees":
                        numEntrees = int(value)
                        self.numEntrees = numEntrees
                        print(numEntrees)

                    elif key == "sortiespotentielles": 
                        self.sortiesPotentielle = eval(value)
                        print("s_Pot = ",self.sortiesPotentielle)
                    elif key == "matrice_de_representants":
                        self.matriceRep = np.fromstring(value,dtype=np.float64, sep=",").reshape(self.k * len(self.sortiesPotentielle), self.numEntrees)
                        print("m_rep = ", self.matriceRep)
        else:
            self.matriceRep = np.empty((k*len(self.sortiesPotentielle), numEntrees))
            self.creerProto(fichierReps)
        self.maxEpoch = 20

    #Fonction pour saisir et créer la matrice des prototypes
    def creerProto(self, fichier):
        f = open(fichier, 'r')
        data = f.read()
        datas = data.split("\n")

        nb_data = len(datas)
        sorties = [datas[i].split(":")[0] for i in range(nb_data)]
        entrees = [(datas[i].split(":")[1]).split(" ") for i in range(nb_data)]
        entrees = [list(filter(None, entree)) for entree in entrees]
        entrees = np.array(entrees)
        entrees = entrees.astype(float)

        #print(entrees[4])

        for k in range(self.k):
             for i, symbol in enumerate(self.sortiesPotentielle):
                 index = sorties.index(str(symbol))
                 self.matriceRep[k*len(self.sortiesPotentielle) + i] = entrees[index][0:self.numEntrees]
                 print(symbol,i, index, entrees[index])

                 #print(sorties)
                 del sorties[index]
                 entrees = np.delete(entrees, index, axis=0)
        


        return entrees, sorties

    #Fonction pour exécuter l'entrainement
    def entraine(self, entree, sortieDesire, ajoutDeDonnees = False, varierEta = False):
        #Premieremnt, nous allons tester si les tableaux entree et sortieDesire contienent des sous tableaux
        #sinon, on les force dans un tableau
        if type(entree[0]) is not list and type(entree[0]) is not np.ndarray: 
            entree = [entree]
            tailleEntree = len(entree)
        entree = list(entree)
        print(entree)
        print("matric de rep : " , self.matriceRep)

        if ajoutDeDonnees:  
            entreeNoisy = self.ajoutDeBruit(entree)
            entree = np.concatenate((entree, entreeNoisy))
            sortieDesire = np.concatenate((sortieDesire, sortieDesire))
            print(entree.shape)
            print(len(sortieDesire))

        permanantEntree = entree
        permanantSortieDesire = sortieDesire

        tailleEntree = len(entree)
        tailleSortie = len(sortieDesire)
        print(tailleEntree,tailleSortie)

        if tailleEntree != tailleSortie:
            raise("len(entree) != len(sortieDesire) : Chaque entree doit avoir une sortie desire conrespondante ")
        for numEpoche , epoche in enumerate(range(self.epoche)): 
            print("epoche " ,self.totalNumEpoche)
            self.performance = np.append(self.performance, [0])
            entree = permanantEntree 
            
            sortieDesire = permanantSortieDesire 

            for i in range(len(entree)):    
                print("entree : ", i)
                
                index = random.randint(0,len(list(entree))-1)
                _entree = entree[index][0:self.numEntrees]
                _sortieDesire = sortieDesire[index]
                entree = np.delete(entree, index,0)
                sortieDesire = np.delete(sortieDesire,index,0)
                
                #Etape 1: Calcul distance 
                minDist =  float("inf")
                minJ = 0
                repPred = 0
                for j in range(len(self.matriceRep)):
                    dist = distance.euclidean(_entree, self.matriceRep[j])
                    if minDist > dist:
                        minDist = dist
                        minJ = j
                repPred =  minJ % len(self.sortiesPotentielle) #Representant predit
                repPred = self.sortiesPotentielle[repPred]
                #print("Entree : ", _entree)
                print("minDIST ", minDist)
                #Etap 2 : Mise a jour des representants
                print(repPred, _sortieDesire)
                if repPred == _sortieDesire: #Si ca correspond, on rapproche
                    self.matriceRep[minJ] = self.matriceRep[minJ] + self.eta * (_entree - self.matriceRep[minJ])
                    self.performance[self.totalNumEpoche] += 1
                    #print("performance ", self.performance)
                else: #Sinon, on eloigne
                    self.matriceRep[minJ] = self.matriceRep[minJ] - self.eta * (_entree - self.matriceRep[minJ])


                #print("Poids Nouveau : ", self.matriceRep[minJ])

            #Fin de l'epoche
            print("Fin de l'eopche")

            #Calucl de la performance avec les donnees de test et avec la validation croisee.
            self.performance[self.totalNumEpoche] = self.performance[self.totalNumEpoche]/tailleEntree
            print("Calcul performance")
            if self.VCin is not None and self.VCout is not None:
                perfVC = self.test(self.VCin, self.VCout)
                self.performanceVC = np.append(self.performanceVC, [perfVC])
            if self.etaAdaptif:
                 self.eta = self.etaInit * (1 - (self.totalNumEpoche/self.maxEpoch))
                 print("Eta changed")
            print("NumEpoche ++")
            self.totalNumEpoche += 1
            print("NumEpoche ++ Done ")

        #print("VC performance apres entrainement: ", self.performanceVC[-1])



    def exporterLVQ(self, fichier):
        print("EXPORTING !!" )
        repertoire = os.path.dirname(fichier)
        if not os.path.exists(repertoire):
            os.makedirs(repertoire)
        f=open(fichier, "w+")
        f.write("Nb_entrees= %d\n" % (self.numEntrees))
        f.write("k= %s\n" % (str(self.k)))
        f.write("eta= %s\n" % (str(self.etaInit)))
        f.write("sortiesPotentielles= %s\n" % (str(self.sortiesPotentielle)))
        np.set_printoptions(threshold = np.prod(self.matriceRep.shape))
        f.write("matrice_de_representants= %s\n" % (np.array2string(self.matriceRep.ravel(), separator = ",").replace("\n", " ").replace("[","").replace("]", "")))
        #print(str(self.matriceRep.ravel()))
        np.set_printoptions(threshold = (1000,1000))

        f.close()
    



    def test(self, entrees, sortieDesire = None):
        '''
        test(entrees, sortieDesire = None)

        Tester le MLP et predire le resultat d'une entree

        '''
        #Les neurones de la premiere couche cache vont prendre les entree du MLP comme entrees 
        #Etape 1 : Activation des Neurons
        print("Starting Test")
        LVQ_out = np.array([])

        if type(entrees[0]) is not list and type(entrees[0]) is not np.ndarray:
            entrees = [entrees]
            tailleEntree = len(entrees)
        else:
            entrees = list(entrees)

        resultats = []
        performance = 0
        for i, entree in enumerate(entrees):
            minDist  = float("inf")

            for j in range(len(self.matriceRep)):
                dist = distance.euclidean(entree[0:self.numEntrees], self.matriceRep[j])
                if minDist > dist:
                    minDist = dist
                    minJ = j
            prediction =  minJ % len(self.sortiesPotentielle) #Representant predit
            prediction = self.sortiesPotentielle[prediction]
                #print(prediction, sortieDesire[i])

            LVQ_out = np.append(LVQ_out,[prediction], axis=0) 

            #resultats.append(MLP_out)
            '''if sortieDesire != None:
                if prediction == sortieDesire[i]:
                    performance += 1'''
        print("Fin du Test")
        if sortieDesire != None:
            performance = np.sum(LVQ_out == sortieDesire)
        return LVQ_out, performance/len(entrees)

    def getMeilleurSortie(self, sortie):
        meilleurSortie = 0
        minDifference = 0
        for sortieDecimal, sortieEncode in self.sortiesPotentielle.items():
            difference = np.abs(sortie - sortieEncode)
            if np.sum(minDifference) < np.sum(difference):
                minDifference = difference
                meilleurSortie = sortieDecimal

        return meilleurSortie

    def softmax(self, X, prob=False):
        softmax_prob  = (np.e**X)/np.sum(np.e**X) 
        softmax_bin = np.zeros(len(X))
        if prob:
            return softmax_prob
        softmax_bin[np.argmax(softmax_prob)] = 1
        return softmax_bin
    
    def ajoutDeBruit(self, entree):
        #http://mms.etsmtl.ca/ELE778/Synthese/8-OptimizationforTrainingDeepModels.pdf
        print("Ajout de bruit")
        noisyArray =  np.random.uniform(0.8,  1.2, [len(entree), entree[0].size])
        print("Noisy Array generated")
        return  np.multiply(noisyArray, entree)

        

def getES(fichier):
    f = open(fichier, 'r')
    data = f.read()
    datas = data.split("\n")
    nb_data = len(datas)
    sorties = [datas[i].split(":")[0] for i in range(nb_data)]
    entrees = [(datas[i].split(":")[1]).split(" ") for i in range(nb_data)]
    print(len(entrees))
    entrees = [list(filter(None, entree)) for entree in entrees]
    entrees = np.array(entrees)
    entrees = entrees.astype(float)

    return entrees, sorties


if __name__ == "__main__":
    sortiesPotentielle = ["o","1", "2", "3", "4", "5","6", "7","8","9"]
    entreeIn, entreeOut = getES("data/data_train.txt")
    VCin, VCout = getES("data/data_vc.txt")
    print(VCin)
    k = 21
    lvq = LVQ(numEntrees=26*60, k = k, sortiePotentielle=sortiesPotentielle,epoche=5, eta=0.1, fichierReps = "data/data_train.txt")

    #print(entreeOut)
    print(len(VCin), len(VCout))
    lvq.entraine(entree=entreeIn[:k*10], sortieDesire=entreeOut[:k*10], varierEta=True, ajoutDeDonnees=True)
    lvq.entraine(entree=entreeIn[:k*10], sortieDesire=entreeOut[:k*10], varierEta=True, ajoutDeDonnees=True)

    lvq.entraine(entree=entreeIn[k*10:], sortieDesire=entreeOut[k*10:], varierEta=True, ajoutDeDonnees=True)


    print("performance :", lvq.performance )

    print("starting TEST")
    _, perf = lvq.test(VCin, sortieDesire = VCout)
    print(perf)
    
    #print(lvq.matriceRep)

    #TEST  (Passed)

    '''sortiesPotentielle = ["1","2"]
    entreeIn, entreeOut = getES("data/data_train_bidon.txt")
    print(entreeIn)
    print(entreeOut)


    lvq = LVQ(numEntrees=4, k = 1, sortiePotentielle=sortiesPotentielle,epoche=1, eta=0.1)
    lvq.entraine(entree=entreeIn, sortieDesire=entreeOut) '''