# Lab 3 : LVQ

One Paragraph of project description goes here

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

Vous êtes maintenant prêt à entrainer le LVQ

## Entrainer un LVQ

1. Clickez sur **Data Entraine** pour selectionner votre fichier texte contenant les données à utiliser pour l'apprentissage
2. Clickez sur **Data VC** pour selectionner votre fichier texte contenant les données à utiliser pour la validation croisé
3. Spécifiez votre **Taux d'apprentissage (η)** 
4. Spécifiez votre **Set de donnée** 
5. Spécifiez le **Nombre d'époches** à faire pour l'entrainement
6. Spécifiez la **Classe de l’unité de sortie**
7. Spécifiez le **k**
8. Spécifiez le **temps maximal** en secondes. Si le temps d'entrainent dépasse **tMax**, le programme arrête l'entrainement
9. Si vous voulez variez le **Taux d'apprentissage (n)**  à chaque époches, cochez **η adaptif**.
10. Si vous voulez doubler le nombre de données d'entrainement en ajoutant du bruit au données qui nous on était fournit, cochez **Ajout de bruit**  

Pour démarer l'entraimenet, clickez sur le boutton **Entraine**

**La courbe bleu represente la performance avec les données d'entrainement et la courbe rouge represente la performance de la validation croisée.**

### Tester un LVQ (Generalisation)

1. Clickez sur **Data Test** pour selectionner votre fichier texte contenant les données à utiliser pour le test
2. Clickez sur le boutton test pour démarer le test.
3. Une fois completé, les résulats de test ainsi que la performance devrait apparaître dans la boite à message

### Sauvgarder un LVQ

Si fois que le LVQ à été entrainé, vous pouvez le sauvgarder.

1. Clicker sur la disquette
2. Donnez un nom à ton LVQ
3. Le fichiez LVQ sera sauvgardé dans ton repertoire de projet dans le dossier : **lvqs_sauvgarde**

**Pour creer un nouveau MLP, vous n'avez qu'a rafraichir la page  **


### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
