#### Plan pour la phase 2 versionning data & modèles
Produire n scipts pour une pipeline de préparation des données, sélection des meilleurs paramètres, entrainement et évaluation



1. Modèle pour les données texte (title produit):
    * script 1 pour split des données (train-test)
    * script 2 pour feature extraction: transformer le texte en matrice creuse (TFIDF)
    * script 3 pour sélection des paramètres: pseudo grid search
    * script 4 pour évaluer le modèle

2. Modèle pour les images:
Problème: modèle très long à faire tourner sur le corpus entier: mini 10h sur gpu

3. Conception des tests unitaires (modèle texte):
En plus des scripts ci dessus:
> * script check status (api en ligne)
> * script inférence sur un datapoint
> * ajouter une fonction d'authentification ?
    
