#######################################################
##  Fichier : ELE767_mlp_lib.py
##  Auteurs : G. Cavero , L. Frija, S. Mohammed
##  Date de creation : 25 Fev 2019          
##  Description : Ce fichier contient toutes les
##                fonctions qui servent a entrainner  
##                un RN MLP comme on a vu en classe
##
##
#######################################################


#from couche2 import Couche
from couche import Couche
import logging, sys
import numpy as np
import random
import re
import os
from scipy.spatial import distance
debug = True


class LVQ(object):
    
    def __init__(self, numEntrees = None, 
                eta = 0.1, sortiePotentielle = None, 
                epoche = 1, etaAdaptif = False, perf_VC = 0.75, 
                VCin = None, VCout = None, fichier_lvq = None, k = 1):   

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

        self.matriceRep = np.empty((k*10, numEntrees))
        self.k = k

        self.creerProto("data/data_train.txt")


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
                 self.matriceRep[k*len(self.sortiesPotentielle) + i] = entrees[index]
                 #print(sorties)
                 del sorties[index]
                 entrees = np.delete(entrees, index, axis=0)


        return entrees, sorties

       
    def entraine(self, entree, sortieDesire, ajoutDeDonnees = False, varierEta = False):
        #Premieremnt, nous allons tester si les tableaux entree et sortieDesire contienent des sous tableaux
        #sinon, on les force dans un tableau
        if type(entree[0]) is not list and type(entree[0]) is not np.ndarray: 
            entree = [entree]
            tailleEntree = len(entree)



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
            self.performance = np.append(self.performance, [0])
            print("epoche " ,self.totalNumEpoche)
            entree = permanantEntree 
            
            sortieDesire = permanantSortieDesire 

            for i in range(len(entree)):    
                #print("entree ", i)
                
                index = random.randint(0,len(list(entree))-1)
                _entree = entree[index][0:self.numEntrees]
                _sortieDesire = sortieDesire[index]
                entree = np.delete(entree, index,0)
                sortieDesire = np.delete(sortieDesire,index,0)
                
                #Etape 1: Calcul distance 
                minDist=  float("inf")
                minJ = 0
                repPred = 0
                for j in range(len(self.matriceRep)):
                    dist = distance.euclidean(_entree, self.matriceRep[j])
                    if minDist > dist:
                        minDist = dist
                        minJ = j
                        repPred =  j % len(self.sortiesPotentielle) #Representant predit
                        repPred = self.sortiesPotentielle[repPred]
                
                #Etap 2 : Actualisation des poids

                print(repPred, _sortieDesire)

                if repPred == _sortieDesire: #Si ca correspond, on rapproche
                    self.matriceRep[minJ] = self.matriceRep[minJ] + self.eta * (_entree - self.matriceRep[minJ])
                    self.performance[self.totalNumEpoche] += 1
                    print("performance ", self.performance)
                else: #Sinon, on eloigne
                    self.matriceRep[minJ] = self.matriceRep[minJ] - self.eta * (_entree - self.matriceRep[minJ])

            #Fin de l'epoche

            #Calucl de la performance avec les donnees de test et avec la validation croisee.
            self.performance[self.totalNumEpoche] = self.performance[self.totalNumEpoche]/tailleEntree
            if self.VCin is not None and self.VCout is not None:
                perfVC = self.test(self.VCin, self.VCout)
                self.performanceVC = np.append(self.performanceVC, [perfVC])
            if self.etaAdaptif:
                 self.eta = self.etaInit * 0.2 ** (self.totalNumEpoche)
                 print("Eta changed")
            self.totalNumEpoche += 1

        print(self.performance[-1])


   

    def exporterLVQ(self, fichier):
        repertoire = os.path.dirname(fichier)
        if not os.path.exists(repertoire):
            os.makedirs(repertoire)
        f=open(fichier, "w+")
        f.write("Nb_neurones_par_CC= %s\n" % (str(self.neuronesParCC)))
        f.write("Nb_entrees= %d\n" % (self.numEntrees))
        f.write("Nb_sorties= %d\n" % (self.numSorties))
        f.write("sortiesPotentielles= %s\n" % (str(self.sortiesPotentielle)))
    

        for i, couche in enumerate(self.couches):
            #f.write("couche #%d \n" % (i))
            f.write("S(%d) = %s \n" % (i, str(list(couche.seuils))))
            #f.write("Neurone#%d \n" % (neurone))
            for src in range(np.shape(couche.poids)[0]):
                for dst in range(np.shape(couche.poids)[1]):
                    f.write("W(%d,%d,%d) =%s \n" % (i,src,dst,str(couche.poids[src ,dst])))
        f.close()
    



    def test(self, entrees, sortieDesire = None):

        '''
        test(entrees, sortieDesire = None)

        Tester le MLP et predire le resultat d'une entree

        '''
        #Les neurones de la premiere couche cache vont prendre les entree du MLP comme entrees 
        #Etape 1 : Activation des Neurons
        MLP_out = np.empty((0,self.numSorties), int)
        if type(entrees[0]) is not list and type(entrees[0]) is not np.ndarray:
            entrees = [entrees]
            tailleEntree = len(entrees)

        if type(sortieDesire[0]) is not list and type(sortieDesire[0]) is not np.ndarray:
            sortieDesire = [sortieDesire]
            tailleSortie = len(sortieDesire)
        resultats = []
        performance = 0
        for i, entree in enumerate(entrees):
            #print("Test Sortie Desire: " + str(i)+ " : "+ str(sortieDesire[i]))
            x = entree[0:self.numEntrees]
            for (j,couche) in enumerate(self.couches):
                couche.setEntrees(x)
                couche.calculSorties()
                x = np.array(couche.sorties)
            #print(MLP_out)
            nn_output = self.softmax(self.couches[-1].getSortie())
            print(MLP_out.size)
            print(nn_output.size)

            MLP_out = np.append(MLP_out,[nn_output], axis=0) 
            print(MLP_out[-1].size)

            #resultats.append(MLP_out)
            if sortieDesire != None:
                if (MLP_out[-1] == sortieDesire[i]).all():
                    performance += 1
        return MLP_out, performance/len(entrees)

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
    entrees = [list(filter(None, entree)) for entree in entrees]
    entrees = np.array(entrees)
    entrees = entrees.astype(float)

    return entrees, sorties


if __name__ == "__main__":
    sortiesPotentielle = ["o","1", "2", "3", "4", "5","6", "7","8","9"]
    lvq = LVQ(numEntrees=1560, k = 10, sortiePotentielle=sortiesPotentielle,epoche=2)
    entreeIn, entreeOut = getES("data/data_train.txt")
    print(len(entreeIn), len(entreeOut))
    lvq.entraine(entree=entreeIn, sortieDesire=entreeOut)
    print("performance :", lvq.performance )
    
    print(lvq.matriceRep)
