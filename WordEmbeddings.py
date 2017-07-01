import numpy as np
from gensim.models import Word2Vec
from pathlib import Path

# Word embeddings
class Word2VecEmbeddings:
    """Create or load word embeddings"""
    def __init__(self,
                 preprocessor, preprocessed_dataset,
                 word_embedding_dimensions, embedding_corpus_name):
        print("Generating word embeddings")
        word_embedding_model = None

        if embedding_corpus_name:
            pretrained_file = Path(embedding_corpus_name)
            if pretrained_file.is_file():
                word_embedding_model = Word2Vec.load(embedding_corpus_name)
                print("Word embeddings loaded!")

        if not word_embedding_model:
            min_word_occurrence = preprocessor.vocabulary.min_word_occurrence if hasattr(preprocessor.vocabulary, "min_word_occurrence") else 5
            word_embedding_model = Word2Vec(preprocessed_dataset.all_preprocessed_tweets(), # preprocessed_dataset.all_tokenized_tweets(),
                                            size=word_embedding_dimensions,
                                            window=7,
                                            min_count=min_word_occurrence,
                                            workers=8,
                                            sg=1,
                                            iter=10)

            print("Word embeddings generated!")

        if embedding_corpus_name:
            word_embedding_model.save(embedding_corpus_name)

        self.embedding_matrix = np.zeros((preprocessor.vocabulary.word_count, word_embedding_dimensions))
        for word, id in preprocessor.vocabulary.word_to_id.items():
            if word in word_embedding_model.wv.vocab:
                embedding_vector = word_embedding_model[word]
                self.embedding_matrix[id] = embedding_vector


# TODO: Implement this
class GloVeEmbeddings:
    pass
