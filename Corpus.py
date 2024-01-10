import re
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from Author import Author

# Définition de la classe Corpus pour gérer un ensemble de documents
class Corpus:
    # Constructeur de la classe
    def __init__(self, nom):
        self.nom = nom  # Nom du corpus
        self.documents = []  # Liste pour stocker les documents
        self.auteurs = {}  # Dictionnaire pour les auteurs et leurs documents
        self.texte_concatene = None  # Texte concaténé de tous les documents pour certaines analyses
        self.freq_table = None  # Table des fréquences des mots
        self.vocab = {}  # Vocabulaire du corpus pour TF et TFxIDF
        self.tf_matrix = None  # Matrice Term Frequency
        self.tfidf_matrix = None  # Matrice TFxIDF

    # Méthode pour ajouter un document au corpus
    def add(self, document):
        self.documents.append(document)  # Ajout du document à la liste
        # Gestion des auteurs
        if document.auteur not in self.auteurs:
            self.auteurs[document.auteur] = Author(document.auteur)
        self.auteurs[document.auteur].add_doc(document)

    # Méthode pour afficher les auteurs et le nombre de leurs documents
    def afficher_auteurs(self):
        for auteur, obj_auteur in self.auteurs.items():
            print(f"{auteur} a publié {obj_auteur.ndoc} documents.")

    # Méthode pour construire le texte concaténé de tous les documents
    def construire_texte_concatene(self):
        if not self.texte_concatene:
            self.texte_concatene = ' '.join([doc.texte for doc in self.documents])
        return self.texte_concatene

    # Méthode pour la recherche d'un mot-clé dans le corpus
    def search(self, mot_cle):
        texte = self.construire_texte_concatene()
        return re.findall(fr'\b{mot_cle}\b', texte, re.IGNORECASE)

    # Méthode pour créer un concordancier
    def concorde(self, expression, taille_contexte=5):
        texte = self.construire_texte_concatene()
        pattern = fr'(.{{0,{taille_contexte}}})({expression})(.{{0,{taille_contexte}}})'
        concordances = re.findall(pattern, texte, re.IGNORECASE)
        return pd.DataFrame(concordances, columns=['Contexte Gauche', 'Motif Trouvé', 'Contexte Droit'])

    # Méthode pour nettoyer le texte (minuscules, suppression de la ponctuation, etc.)
    def nettoyer_texte(self, texte):
        texte = texte.lower()
        texte = re.sub(r'\n', ' ', texte)
        texte = re.sub(r'[^\w\s]', '', texte)
        return texte

    # Méthode pour afficher les statistiques textuelles
    def stats(self, n=10):
        if not self.freq_table:
            self.freq_table = self.construire_freq_table()
        print(f"Nombre de mots différents: {len(self.freq_table)}")
        print("Mots les plus fréquents:")
        print(self.freq_table.head(n))

    # Méthode pour construire la table des fréquences des mots
    def construire_freq_table(self):
        vocabulaire = set()
        for doc in self.documents:
            texte_nettoye = self.nettoyer_texte(doc.texte)
            mots = texte_nettoye.split()
            vocabulaire.update(mots)

        freq_table = pd.DataFrame(index=list(vocabulaire), columns=['Fréquence', 'Doc Frequency'])
        for mot in vocabulaire:
            freq_table.loc[mot, 'Fréquence'] = sum(doc.texte.count(mot) for doc in self.documents)
            freq_table.loc[mot, 'Doc Frequency'] = sum(mot in doc.texte for doc in self.documents)

        return freq_table.sort_values(by='Fréquence', ascending=False)

    # Méthode pour construire le vocabulaire et la matrice TF
    def construire_vocab_tf(self):
        word_id = 0
        for doc in self.documents:
            words = re.findall(r'\b\w+\b', doc.texte.lower())
            for word in set(words):
                if word not in self.vocab:
                    self.vocab[word] = {'id': word_id, 'doc_count': 0, 'total_occurrences': 0}
                    word_id += 1
                self.vocab[word]['total_occurrences'] += words.count(word)
                self.vocab[word]['doc_count'] += 1

    # Méthode pour construire la matrice TF (Term Frequency)
    def construire_tf_matrix(self):
        rows, cols, data = [], [], []
        for doc_id, doc in enumerate(self.documents):
            words = re.findall(r'\b\w+\b', doc.texte.lower())
            for word in set(words):
                word_id = self.vocab[word]['id']
                count = words.count(word)
                rows.append(doc_id)
                cols.append(word_id)
                data.append(count)
        self.tf_matrix = csr_matrix((data, (rows, cols)), shape=(len(self.documents), len(self.vocab)))

    # Méthode pour construire la matrice TFxIDF (Term Frequency-Inverse Document Frequency)
    def construire_tfidf_matrix(self):
        transformer = TfidfTransformer()
        self.tfidf_matrix = transformer.fit_transform(self.tf_matrix)

    # Méthode pour créer un moteur de recherche dans le corpus
    def moteur_recherche(self, query):
        query_vector = self.create_query_vector(query)
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix)

        scores = cosine_similarities.flatten()
        sorted_doc_indices = np.argsort(scores)[::-1]  # Tri par ordre décroissant

        top_results = 5  # Nombre de résultats à afficher
        top_doc_indices = sorted_doc_indices[:top_results]

        results = [(self.documents[idx].titre, scores[idx]) for idx in top_doc_indices if scores[idx] > 0]
        return results

    # Méthode pour créer un vecteur de requête basé sur les mots-clés de la requête
    def create_query_vector(self, query):
        query_words = re.findall(r'\b\w+\b', query.lower())
        query_vector = np.zeros((1, len(self.vocab)))

        for word in query_words:
            if word in self.vocab:
                word_id = self.vocab[word]['id']
                query_vector[0, word_id] = 1

        transformer = TfidfTransformer()
        return transformer.fit_transform(query_vector)

    # Méthode spéciale pour afficher une représentation en chaîne de caractères du corpus
    def __str__(self):
        return f"Corpus '{self.nom}' contenant {len(self.documents)} documents et {len(self.auteurs)} auteurs."
