# Importation des bibliothèques nécessaires
import praw
import urllib.request
import xmltodict
from datetime import datetime

# Importation des classes depuis leurs fichiers respectifs
from Document import Document, RedditDocument, ArxivDocument
from Corpus import Corpus
from Author import Author

# Initialisation du client Reddit
reddit = praw.Reddit(client_id='4SMSDNTuqNZR5xvt5fxpiA',
                     client_secret='sfetYZc1mZ6BKX4fhZoP1fN78PH2Wg',
                     user_agent='TD script par Alireza268')

# Récupération des posts Reddit
subreddit = reddit.subreddit('Coronavirus')
limit = 100
textes_Reddit = []

for post in subreddit.hot(limit=limit):
    titre = post.title.replace("\n", " ")
    auteur = post.author.name if post.author else "Inconnu"
    date = datetime.fromtimestamp(post.created_utc)
    url = post.url
    nb_commentaires = post.num_comments
    texte = post.selftext if post.selftext else titre
    reddit_doc = RedditDocument(titre, auteur, date, url, texte, nb_commentaires)
    textes_Reddit.append(reddit_doc)

# Récupération des articles Arxiv
url_arxiv = 'http://export.arxiv.org/api/query?search_query=all:covid&start=0&max_results=100'
data = urllib.request.urlopen(url_arxiv).read()
data_dict = xmltodict.parse(data)

textes_Arxiv = []

for entry in data_dict['feed']['entry']:
    titre = entry['title']
    # Vérification si 'authors' existe et a la clé 'author'
    if 'authors' in entry and 'author' in entry['authors']:
        auteur = ', '.join([author['name'] for author in entry['authors']['author']])
    else:
        auteur = "Inconnu"
    date = datetime.strptime(entry['published'], '%Y-%m-%dT%H:%M:%SZ')
    url = entry['id']
    resume = entry['summary'] if 'summary' in entry else ""
    arxiv_doc = ArxivDocument(titre, auteur, date, url, resume, [])
    textes_Arxiv.append(arxiv_doc)

# Initialisation et ajout des documents au Corpus
corpus = Corpus("MonCorpus")
for doc in textes_Reddit + textes_Arxiv:
    corpus.add(doc)

# Construction des matrices TF et TFxIDF
corpus.construire_vocab_tf()
corpus.construire_tf_matrix()
corpus.construire_tfidf_matrix()

# ========== Tests Approfondis ==========

# Test 1: Vérification du moteur de recherche avec une requête différente
print("\n=== Test 1: Moteur de Recherche avec 'flu' ===")
query = "flu"
print(f"\nRésultats de recherche pour '{query}':")
search_results_flu = corpus.moteur_recherche(query)
for titre, score in search_results_flu[:10]:  # Affiche les 10 premiers résultats
    print(f"{titre} (Score: {score:.2f})")

# Test 2: Moteur de recherche avec une requête "vaccine"
print("\n=== Test 2: Moteur de Recherche avec 'vaccine' ===")
query = "vaccine"
print(f"\nRésultats de recherche pour '{query}':")
search_results_vaccine = corpus.moteur_recherche(query)
for titre, score in search_results_vaccine[:10]:
    print(f"{titre} (Score: {score:.2f})")

# Test 3: Concordancier pour le terme "treatment"
print("\n=== Test 3: Concordancier pour 'treatment' ===")
expression = "treatment"
print(f"\nConcordancier pour '{expression}':")
concordancier_treatment = corpus.concorde(expression, taille_contexte=10)
print(concordancier_treatment.head())

# Test 4: Moteur de recherche avec une expression composée "social distancing"
print("\n=== Test 4: Moteur de Recherche avec 'social distancing' ===")
query = "social distancing"
print(f"\nRésultats de recherche pour '{query}':")
search_results_social_distancing = corpus.moteur_recherche(query)
for titre, score in search_results_social_distancing[:5]:
    print(f"{titre} (Score: {score:.2f})")

# Test 5: Affichage de la matrice TF
print("\n=== Test 4: Matrice TF ===")
tf_matrix_dense = corpus.tf_matrix.todense()
print("Un échantillon de la matrice TF:")
print(tf_matrix_dense[:5, :5])  # Affiche les 5 premières lignes et colonnes de la matrice TF

# Test 6: Affichage de la matrice TFxIDF
print("\n=== Test 5: Matrice TFxIDF ===")
tfidf_matrix_dense = corpus.tfidf_matrix.todense()
print("Un échantillon de la matrice TFxIDF:")
print(tfidf_matrix_dense[:5, :5])  # Affiche les 5 premières lignes et colonnes de la matrice TFxIDF

# Test 7: Validation du contenu des documents
print("\n=== Test 6: Contenu des Documents ===")
for i, doc in enumerate(corpus.documents[:5]):
    print(f"\nDocument {i+1} Titre: {doc.titre}")
    print(f"Texte: {doc.texte[:100]}...")  # Affiche les 100 premiers caractères du texte

# ========== Affichage des Informations du Corpus ==========
print(f"\nNombre total de documents dans le corpus : {len(corpus.documents)}")

# Affichage d'un échantillon de documents
print("\nÉchantillon de documents du corpus:")
for i, doc in enumerate(corpus.documents[:5]):
    print(f"Document {i+1}: {doc}")

# Affichage des auteurs
print("\nListe des auteurs dans le corpus:")
corpus.afficher_auteurs()
