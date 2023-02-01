Voici les étapes pour utiliser Apache Airflow pour planifier et exécuter ce script sur un environnement local :

    Installation d'Apache Airflow : Vous pouvez l'installer en utilisant pip en tapant la commande suivante :
    pip install apache-airflow

    Initialisation de l'environnement Airflow : Après l'installation, vous devez initialiser l'environnement Airflow en exécutant la commande suivante :
```
    airflow initdb
```

Configuration des tâches : Pour créer une tâche DAG (Directed Acyclic Graph), création d'un fichier Python airflow.py et configuration de la tâche DAG pour planifier l'exécution du script toutes les 10 minutes en utilisant les paramètres de schedule_interval et start_date.

    voir le script que j'ai fait airflow.py


   Démarrage du serveur d'airflow : Pour démarrer le serveur Airflow, vous pouvez exécuter la commande suivante :
```
    airflow webserver -p 8080
```
    Exécution de la tâche : Pour exécuter la tâche, vous pouvez utiliser l'interface web d'Airflow en accédant à l'adresse suivante :
```
    http://localhost:8080/
```