# Importation des bibliothèques nécessaires
import praw
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import urllib.request
import xmltodict
from datetime import datetime

# Importation des classes depuis leurs fichiers respectifs
from Corpus import Corpus
from Document import RedditDocument, ArxivDocument

# Initialisation du client Reddit avec les identifiants de l'API
reddit = praw.Reddit(client_id='4SMSDNTuqNZR5xvt5fxpiA',
                     client_secret='sfetYZc1mZ6BKX4fhZoP1fN78PH2Wg',
                     user_agent='TD script par Alireza268')

# Initialisation du corpus et chargement des données
corpus = Corpus("MonCorpus")
subreddit = reddit.subreddit('Coronavirus')

# Récupération des posts Reddit
for post in subreddit.hot(limit=100):
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
                auteur = ', '.join([author['name'] for author in entry['authors']['author']])
            else:
                auteur = entry['authors']['author']['name']
        else:
            auteur = "Inconnu"
        date = datetime.strptime(entry['published'], '%Y-%m-%dT%H:%M:%SZ')
        url = entry['id']
        resume = entry['summary'] if 'summary' in entry else ""
        arxiv_doc = ArxivDocument(titre, auteur, date, url, resume, [])
        corpus.add(arxiv_doc)

# Construction des matrices TF et TFxIDF
corpus.construire_vocab_tf()
corpus.construire_tf_matrix()
corpus.construire_tfidf_matrix()

# Initialiser l'application Dash
app = dash.Dash(__name__)

# Layout de l'application
app.layout = html.Div([
    dcc.Input(id='input-text', type='text', placeholder='Entrez un mot-clé...'),
    html.Button('Rechercher', id='button-search'),
    html.Div(id='search-results')
])

# Callback pour la recherche
@app.callback(
    Output('search-results', 'children'),
    [Input('button-search', 'n_clicks')],
    [dash.dependencies.State('input-text', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks and value:
        search_results = corpus.moteur_recherche(value)
        return html.Ul(children=[html.Li(f"{titre} (Score: {score:.2f})") for titre, score in search_results])
    return html.Div()

# Exécuter l'application
if __name__ == '__main__':
    app.run_server(debug=True)
