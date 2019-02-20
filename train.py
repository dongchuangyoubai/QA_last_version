import tensorflow as tf

from qa_model import QASystem, Encoder, Decoder
from config import Config
from data_utils import *
import os
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [str(line).strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def run_func():
    config = Config()
    train = squad_dataset(config.question_train, config.context_train, config.answer_train)
    dev = squad_dataset(config.question_dev, config.context_dev, config.answer_dev)
    # print(config.question_train)
    embed_path = config.embed_path
    vocab_path = config.vocab_path
    # print(config.embed_path, config.vocab_path)
    vocab, rev_vocab = initialize_vocab(vocab_path)

    embeddings = get_trimmed_glove_vectors(embed_path)

    encoder = Encoder(config.hidden_size)
    decoder = Decoder(config.hidden_size)

    qa = QASystem(encoder, decoder, embeddings, config)

    with tf.Session() as sess:
        # ====== Load a pretrained model if it exists or create a new one if no pretrained available ======
        qa.initialize_model(sess, config.train_dir)
        # train process
        # qa.train(sess, [train, dev], config.train_dir)
        # em = qa.evaluate_model(sess, dev)

        # run process
        while True:
            question = input('please input question: ')
            if question == 'exit':
                break
            raw_context = input('please input context: ')
            if raw_context == 'exit':
                break
            question = [vocab[x] if x in vocab.keys() else 2 for x in question.split()]
            context = [vocab[x] if x in vocab.keys() else 2 for x in raw_context.split()]
            test = [[question], [context], [[1, 2]]]
            a_s, a_e = qa.answer(sess, test)
            if a_e == a_s:
                print("answer: ", raw_context.split()[a_s[0]])
            else:
                print("answer: ", ' '.join(raw_context.split()[a_s[0]: a_e[0] + 1]))


if __name__ == "__main__":
    run_func()
