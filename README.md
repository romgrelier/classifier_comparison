Ce projet nécessite un interpreteur python 3.6, pour faciliter l'installation, un environnement virtuel est utilisé dont les dépendances sont listées dans le fichiers requirement.txt

les commandes à lancer sont :

python3 -m venv env
(cette dernière commande peut échouer si le paquet python3-venv n'est pas installé -> "sudo apt install python3-venv" )
source env/bin/activate
pip3 install -r requirement.txt
python3 run.py