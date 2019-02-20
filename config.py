import os

class Config:
    num_epochs = 100
    batch_size = 16
    train_embeddings = 0
    max_gradient_norm = -1
    hidden_size = 150
    num_layers = 3
    embedding_size = 300
    data_dir = os.path.join('data', 'squad')
    vocab_path = os.path.join(data_dir, "vocab.dat")
    embed_path = os.path.join(data_dir, "glove.trimmed.300.npz")
    dropout_val = 1.0
    train_dir = "models_lstm_basic"
    use_match = 1
    keep_prob = 0.8
    

    def get_paths(mode, data_dir):
        question = os.path.join(data_dir, "%s.ids.question" % mode)
        context = os.path.join(data_dir, "%s.ids.context" % mode)
        answer = os.path.join(data_dir, "%s.span" % mode)

        return question, context, answer 

    question_train, context_train, answer_train = get_paths("train", data_dir)
    question_dev, context_dev, answer_dev = get_paths("val", data_dir)
