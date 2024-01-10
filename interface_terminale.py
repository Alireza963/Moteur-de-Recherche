# Importation des bibliothèques nécessaires
import praw
import urllib.request
import xmltodict
from datetime import datetime

# Importation des classes depuis leurs fichiers respectifs
from Document import Document, RedditDocument, ArxivDocument
from Corpus import Corpus
from Author import Author

# Initialisation du client Reddit avec les identifiants de l'API
reddit = praw.Reddit(client_id='4SMSDNTuqNZR5xvt5fxpiA',
                     client_secret='sfetYZc1mZ6BKX4fhZoP1fN78PH2Wg',
                     user_agent='TD script par Alireza268')

# Définition du subreddit à partir duquel les posts seront récupérés
subreddit = reddit.subreddit('Coronavirus')

# Initialisation et chargement du corpus
corpus = Corpus("MonCorpus")

# Récupération des posts Reddit depuis le subreddit 'Coronavirus'
for post in subreddit.hot(limit=100):
    # Traitement de chaque post pour récupérer les informations pertinentes
    titre = post.title.replace("\n", " ")
    auteur = post.author.name if post.author else "Inconnu"
    date = datetime.fromtimestamp(post.created_utc)
    url = post.url
    nb_commentaires = post.num_comments
    texte = post.selftext if post.selftext else titre
    reddit_doc = RedditDocument(titre, auteur, date, url, texte, nb_commentaires)
    corpus.add(reddit_doc)

# Récupération des articles Arxiv
url_arxiv = 'http://export.arxiv.org/api/query?search_query=all:covid&start=0&max_results=100'
data = urllib.request.urlopen(url_arxiv).read()
data_dict = xmltodict.parse(data)

if 'entry' in data_dict['feed']:
    for entry in data_dict['feed']['entry']:
        titre = entry['title']
        if 'authors' in entry and 'author' in entry['authors']:
            if isinstance(entry['authors']['author'], list):
                # Plusieurs auteurs
                auteur = ', '.join([author['name'] for author in entry['authors']['author']])
            else:
                # Un seul auteur
                auteur = entry['authors']['author']['name']
        else:
            auteur = "Inconnu"

        date = datetime.strptime(entry['published'], '%Y-%m-%dT%H:%M:%SZ')
        url = entry['id']
        resume = entry['summary'] if 'summary' in entry else ""
        arxiv_doc = ArxivDocument(titre, auteur, date, url, resume, [])
        corpus.add(arxiv_doc)


# Construction des matrices TF (Term Frequency) et TFxIDF (Term Frequency-Inverse Document Frequency)
corpus.construire_vocab_tf()
corpus.construire_tf_matrix()
corpus.construire_tfidf_matrix()

# Interface en ligne de commande pour interagir avec le corpus
def main():
    while True:
        print("\n==== Interface de Recherche du Corpus ====")
        query = input("Entrez un mot-clé pour la recherche (ou 'exit' pour quitter): ")
        if query.lower() == 'exit':
            break
        search_results = corpus.moteur_recherche(query)
        if search_results:
            print("\nRésultats de recherche :")
            # Affiche les 10 premiers résultats avec le score le plus élevé
            for titre, score in sorted(search_results, key=lambda x: x[1], reverse=True)[:10]:
                print(f"- {titre} (Score: {score:.2f})")
        else:
            print("Aucun résultat trouvé.")

# Point d'entrée du script
if __name__ == "__main__":
    main()
