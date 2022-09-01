from gensim.models import KeyedVectors, Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

def update_word2vec(vocab, no_header= True, f_name="model_vocab.txt"):
    model_2 = Word2Vec(vector_size=100, min_count=1)
    model_2.build_vocab(vocab)
    total_examples = model_2.corpus_count
    w2v_model = KeyedVectors.load_word2vec_format("model_vocab.txt", no_header=no_header,binary=False)
    model_2.build_vocab([list(w2v_model.key_to_index.keys())], update=True)
    model_2.train(vocab, total_examples=total_examples, epochs=model_2.epochs)
    model_2.wv.save_word2vec_format(f_name, binary=False)