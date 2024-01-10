# Importation des bibliothèques nécessaires
import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Définition de la classe Document, qui sera la base pour tous les types de documents
class Document:
    # Constructeur de la classe, avec les attributs de base pour un document
    def __init__(self, titre: str, auteur: str, date: datetime.date, url: str, texte: str):
        self.titre = titre  # Titre du document
        self.auteur = auteur  # Auteur du document
        self.date = date  # Date de publication du document
        self.url = url  # URL du document
        self.texte = texte  # Contenu textuel du document

    # Méthode spéciale pour afficher une représentation en chaîne de caractères du document
    def __str__(self):
        return f"{self.titre} par {self.auteur}, publié le {self.date}."

    # Méthode pour analyser le texte du document
    def analyser_texte(self):
        mots = word_tokenize(self.texte)  # Tokenisation du texte
        mots_filtrés = [mot for mot in mots if mot not in stopwords.words('french')]  # Filtrage des stopwords
        frequences = Counter(mots_filtrés)  # Comptage des fréquences des mots
        return frequences

# Classe RedditDocument, héritant de Document, spécifique aux documents issus de Reddit
class RedditDocument(Document):
    # Constructeur de la classe, incluant un attribut spécifique pour le nombre de commentaires
    def __init__(self, titre, auteur, date, url, texte, nb_commentaires):
        super().__init__(titre, auteur, date, url, texte)  # Appel du constructeur de la classe parente
        self.nb_commentaires = nb_commentaires  # Nombre de commentaires du post Reddit

# Classe ArxivDocument, héritant de Document, spécifique aux documents issus de Arxiv
class ArxivDocument(Document):
    # Constructeur de la classe, incluant un attribut pour les co-auteurs
    def __init__(self, titre, auteur, date, url, texte, co_auteurs):
        super().__init__(titre, auteur, date, url, texte)  # Appel du constructeur de la classe parente
        self.co_auteurs = co_auteurs  # Liste des co-auteurs du document Arxiv
