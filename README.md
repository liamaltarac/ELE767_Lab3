# ELE767 Lab 3 : LVQ 

## Comment utiliser

Ces instructions vous permeterons d'obtenir une copie du projet et de la rouler sur votre machine. 

### Prerequis

Voici une liste des dépendances à obtenir avant de lancer le programme

* **Python 3** 
* **scipy** (pip install scipy)
* **numpy** (pip install numpy)
* **Flask** (pip install flask)
* Une connection internet pour l'interface.


### Installation

Voici les étapes détaillés à suivre pour installer et lancer le programme

Dans votre ligne de commande, entrez : 

```
git clone https://github.com/liamaltarac/ELE767_Lab3.git
```

Une fois telechargé, entrez dans le repertoire du projet et ecrivez dans votre terminal : 

```
python main.py
```

Ensuite, dans un navigateur web, allez à l'addresse 

```
localhost:5000
```
![](https://media.giphy.com/media/2UwFWYXdqzeMY0Xn79/giphy.gif)

Vous êtes maintenant prêt à entrainer le LVQ

## Entrainer un LVQ

1. Clickez sur **Data Entraine** pour selectionner votre fichier texte contenant les données à utiliser pour l'apprentissage
2. Clickez sur **Data VC** pour selectionner votre fichier texte contenant les données à utiliser pour la validation croisé
3. Spécifiez votre **Taux d'apprentissage (η)** 
4. Spécifiez votre **Set de donnée** 
5. Spécifiez le **Nombre d'époches** à faire pour l'entrainement
6. Spécifiez la **Classe de l’unité de sortie**
7. Spécifiez le **k**
8. Spécifiez le **temps maximal** en secondes. Si le temps d'entrainent dépasse **tMax**, le programme arrête l'entrainement. Si vide, **tMax = infini**
9. Si vous voulez variez le **Taux d'apprentissage (n)**  à chaque époches, cochez **η adaptif**.
10. Si vous voulez doubler le nombre de données d'entrainement en ajoutant du bruit au données qui nous on était fournit, cochez **Ajout de bruit**  

Pour démarer l'entraimenet, clickez sur le boutton **Entraine**

**La courbe bleu represente la performance avec les données d'entrainement et la courbe rouge represente la performance de la validation croisée.**

## Tester un LVQ (Generalisation)

1. Clickez sur **Data Test** pour selectionner votre fichier texte contenant les données à utiliser pour le test
2. Clickez sur le boutton test pour démarer le test.
3. Une fois completé, les résulats de test ainsi que la performance devrait apparaître dans la boite à message

## Sauvgarder un LVQ

Si fois que le LVQ à été entrainé, vous pouvez le sauvgarder.

1. Clicker sur la disquette
2. Donnez un nom à ton LVQ
3. Le fichiez LVQ sera sauvgardé dans ton repertoire de projet dans le dossier : **lvqs_sauvgarde**

## Ouvrir un LVQ

1. Clicker sur l'envloppe
2. Dans le dossier : **lvqs_sauvgarde**, choisisez voitre fichier texte contenant le LVQ
3. Vous pouvez maintenant continuer l'entrainement de votre LVQ et le tester.


**Pour creer un nouveau MLP, vous n'avez qu'a rafraichir la page**

## Autheurs

* **Gabriella Cavero Linares**
* **Liam Frija-Altarac**
* **Saddat Mohammad**
