from flask import Flask,  Response
from flask import request
from flask import render_template
import json
from werkzeug.utils import secure_filename
import os
import numpy as np

from ele767_lvq_lib import LVQ
import webbrowser
import configparser
import time
import traceback

#######################################################
##  Fichier : main.py
##  Auteurs : G. Cavero , L. Frija, S. Mohammed
##  Date de creation : 25 Fev 2019          
##  Description : Ce fichier contient toutes les
##                fonctions qui sont utilisee pour l'interface
##
#######################################################


app = Flask(__name__)
app.lvq = None
#Ouvre l'interface 
'''
def create_app():
    app = Flask(__name__)
    app.lvq = None
    time.sleep(1)  #Attendre. Pour ne pas ouvrir l'interface 2 fois
    def run_on_start(*args, **argv):
        url = "http://localhost:5000"
        webbrowser.open(url, new = 0, autoraise=True)
        print("opening the webbrowser")
    run_on_start()
    return app
app = create_app()  '''
 
UPLOAD_FOLDER = 'UPLOAD_FOLDER'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



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

@app.route('/')
def home():

    if request.method == "GET":
        nb_epoche = None
        db=  None
        eta=None
        n_p_cc =  None
        fct= None
        error = None
        sorties_potentielles = None
        app.lvq = None
        
    
        config = configparser.ConfigParser()
        config.read('default_config.ini')

        nb_epoche = config['DEFAULT']['nb_epoches']
        db = config['DEFAULT']['base_de_donnees']
        eta = config['DEFAULT']['eta']
        classe_sortie = config['DEFAULT']['classe_sortie']
        return render_template("index.html", nb_epoche=nb_epoche, db=str(db), 
                                eta=str(eta), error=error, 
                                sorties_potentielles=str(classe_sortie))
    else:
        return "GOT POST REQUETST"

    

@app.route('/start_training',methods=['POST'])
def startTraining():
    trainingParams = {}
    status = {}

    try:
        if request.method == "POST":
            print(request.form)
            #Nous allons premierement prendre les fichiers d'entrainement et de validation croisé. 
            if 'dataTrain' in request.files:
                file = request.files["dataTrain"]
                trainingParams["data_entrain"] = file
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                dataTrainFile = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            else:
                dataTrainFile = None
            if 'dataVC' in request.files:
                file = request.files['dataVC']
                trainingParams["data_vc"] = file
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                dataVCFile = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            else:
                dataVCFile = None
            #Ensuite, on separe tout nos données de la requete
            print(request.form['eta'])
            eta =  float(request.form['eta'])
            db = int(request.form['db'])
            nb_epoche = int(request.form['nb_epoche'])
            sorties_desire = request.form['sortiesDes']
            ajoutBruit = request.form['ajoutBruit']
            etaAdaptif = request.form['etaAdaptif']
            print(sorties_desire)
            sorties_desire = sorties_desire.replace(" ", "").split(",") # On convertit 'o', '1' ,'2' --> ['o', '1', '2']

            nb_entrees = db * 26
            
            if app.lvq == None:
                print("NEW lvq")
                if etaAdaptif == "True":
                    etaAdaptif = True
                else:
                    etaAdaptif = False

                app.lvq = LVQ(nb_entrees, eta = eta, sortiePotentielle = sorties_desire, 
                            epoche = 1, etaAdaptif=etaAdaptif, k=100)
                print("LVQ cree")
            trainInput, trainOutput = getES(dataTrainFile)

            boolAjoutBruit = False
            if ajoutBruit == "True":
                boolAjoutBruit = True
            app.lvq.entraine(trainInput, trainOutput, boolAjoutBruit)
            print("training DONE")

            if dataVCFile is not None:
                print("Starting VC")

                vcIn, vcOut = getES(dataVCFile)
                _, vcPerf = app.lvq.test(vcIn, sortieDesire =  vcOut)
                status["vcDataPerf"] = vcPerf

            status["status"] = "OK"
            status["trainDataPerf"] = app.lvq.performance[-1]
            status["eta"] = app.lvq.eta

    except Exception as e:
        status["status"] = "FAIL"

    traceback.print_exc()
    
    return json.dumps(status)

@app.route('/open_lvq',methods=['POST'])
def openLVQ():
    #Cette fonction gere l'ouverture d'un lvq existant
    if request.method == "POST":
        print(request.files)
        status = {}
        status["status"] = "OK"

        try:
            if 'lvqFile' in request.files:
                file = request.files["lvqFile"]
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                lvqFile = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                app.lvq = LVQ(fichier_lvq=lvqFile)
                sortiesPotentielle = ""
                for key, val in app.lvq.sortiesPotentielle.items():
                    print("'"+key + "'" + ":" + str(val) + "\n")
                    sortiesPotentielle += ("'"+key + "'" + ":" + str(val) + ",\n")
                status["sortiesPotentielle"] = sortiesPotentielle
                print("numEntrees ", app.lvq.numEntrees )
                status["db"] = app.lvq.numEntrees / 26
                status["eta"] = app.lvq.eta
            
            print("Done" + sortiesPotentielle)
        except Exception as e:
            status["status"] = "ERREUR"
            print(e)

        return json.dumps(status)

@app.route('/save_lvq',methods=['POST'])
def saveLVQ():
    #Cette fonction gere le sauvgard d'un lvq
    if request.method == "POST":
        print(request.form)
        fileOut = os.path.join("lvqs_sauvgarde", request.form["outputFile"] + ".txt")
        status = {}

        if app.lvq is None:
            status["status"] = "Aucun lvq ouvert"
            return json.dumps(status)

        try:
            app.lvq.exporterLVQ(fileOut)
            status["status"] = "Fini : \n\tSauvgardé sous le nom: \n\t "+fileOut

        except Exception as e:
            print(e)    
            status["status"] = "ERREUR"
            
        return json.dumps(status)


@app.route('/test_lvq',methods=['POST'])
def testLVQ():
    if request.method == "POST":
        print(request.files)
        status = {}

        #Nous allons premierement prendre les fichiers d'entrainement et de validation croisé. 
        if 'lvqTestFile' in request.files:
            file = request.files["lvqTestFile"]
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            dataTestFile = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if app.lvq is None:
                status["status"] = "Aucun LVQ ouvert"
                return json.dumps(status)
                
            try:
                print("STARTING TEST")

                print(dataTestFile)
                testOut  = np.empty((0,1), int)
                testIn , testOutDes = getES(dataTestFile)
                print("STARTING TEST")
                #print(testOutDes)
                lvq_out, perf = app.lvq.test(testIn, sortieDesire = testOutDes)
                print("lvq_OUT")
                print(lvq_out)
                #print(app.lvq.sortiesPotentielle)
                #lvq_out = lvq_out.astype(int)
                #sortPotInv = dict([[str(v),k] for k,v in app.lvq.sortiesPotentielle.items()])
                #print(sortPotInv)
                print(len(lvq_out))
                for i in range(len(lvq_out)):
                    print("OUTPUT", lvq_out[i])
                    testOut = np.append(testOut, ["Echantillon " + str(i) + " : " + lvq_out[i]])
                #print(testOut)

                status["status"] = "Fin du test"
                status["lvq_out"] = str(np.vstack(testOut))
                print(perf)
                status["perf"] = perf
            except Exception as e:
                status["status"] = "ERREUR"
                print(e)
 
    return json.dumps(status)

if __name__ == "__main__":
    app.run(debug=True,  threaded=False)
    print("starting")

