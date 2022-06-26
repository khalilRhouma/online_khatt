import os
from configparser import ConfigParser
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from ds_ctcdecoder import Scorer, ctc_beam_search_decoder
from src.features.datasets import handwriting_to_input_vector, pad_sequences
from src.features.feature import calculate_feature_vector_sequence
from src.features.preprocessing import preprocess_handwriting
from src.models.rnn import bi_rnn

# from src.utils.set_dirs import get_conf_dir
from src.utils.text import Alphabet, decodex


@dataclass
class Loader:
    config_file: str
    model_path: str
    lm_binary_path: str
    lm_trie_path: str
    config_header_nn: str = field(init=False)
    config_header_nn: str = field(init=False)
    network_type: str = field(init=False)
    n_context: int = field(init=False)
    n_input: int = field(init=False)
    beam_search_decoder: str = field(init=False)
    # LM setting
    lm_alpha: float = field(init=False)
    lm_beta: float = field(init=False)
    beam_width: int = field(init=False)
    cutoff_prob: float = field(init=False)
    cutoff_top_n: int = field(init=False)

    def __post_init__(self):

        self.config_header_nn = "nn"
        self.config_header_lm = "lm"

        parser = ConfigParser(os.environ)

        self.conf_path = os.path.join("../src/configs", self.config_file)
        parser.read(self.conf_path)

        self.network_type = parser.get(self.config_header_nn, "network_type")
        self.n_context = parser.getint(self.config_header_nn, "n_context")
        self.n_input = parser.getint(self.config_header_nn, "n_input")
        self.beam_search_decoder = parser.get(
            self.config_header_nn, "beam_search_decoder"
        )

        # LM setting
        self.lm_alpha = parser.getfloat(self.config_header_lm, "lm_alpha")
        self.lm_beta = parser.getfloat(self.config_header_lm, "lm_beta")
        self.beam_width = parser.getint(self.config_header_lm, "beam_width")
        self.cutoff_prob = parser.getfloat(self.config_header_lm, "cutoff_prob")
        self.cutoff_top_n = parser.getint(self.config_header_lm, "cutoff_top_n")


def create_tf_session(model_path):
    saver = tf.train.Saver()
    # create the session
    sess = tf.Session()
    saver.restore(sess, model_path)
    print("Model restored")
    return sess


def preprocess_data(points):
    # print("Points before pre",points.shape)
    points = np.array(points)
    norm_args = ["origin", "filp_h", "smooth", "slope", "resample", "slant", "height"]
    feat_args = [
        "x_cor",
        "y_cor",
        "penup",
        "dir",
        "curv",
        "vic_aspect",
        "vic_curl",
        "vic_line",
        "vic_slope",
        "bitmap",
    ]
    # print("Normalizing trajectory...")
    traj = preprocess_handwriting(points, norm_args)
    # print(traj)
    # print("Calculating feature vector sequence...")
    feat_seq_mat = calculate_feature_vector_sequence(traj, feat_args)
    feat_seq_mat = feat_seq_mat.astype("float32")
    feat_seq_mat.shape

    data = []

    train_input = handwriting_to_input_vector(feat_seq_mat, 20, 9)
    train_input = train_input.astype("float32")

    data.append(train_input)
    # data_len

    data = np.asarray(data)
    # data_len = np.asarray(train_input)
    return data


def do_inference(points, config_file, model_path, lm_binary_path, lm_trie_path):

    loader = Loader(
        config_file=config_file,
        model_path=model_path,
        lm_binary_path=lm_binary_path,
        lm_trie_path=lm_trie_path,
    )

    input_tensor = tf.placeholder(
        tf.float32,
        [None, None, loader.n_input + (2 * loader.n_input * loader.n_context)],
        name="input",
    )
    seq_length = tf.placeholder(tf.int32, [None], name="seq_length")
    logits, _ = bi_rnn(
        loader.conf_path,
        input_tensor,
        tf.to_int64(seq_length),
        loader.n_input,
        loader.n_context,
    )
    sess = create_tf_session(loader.model_path)

    data = preprocess_data(points)
    # alphabet = Alphabet('../backwalter_labels.txt')
    alphabet = Alphabet("../alphabet.txt")
    # convert this to funcation
    mapping = {}
    with open("../arabic_mapping.txt", "r", encoding="utf-8") as inf:
        for line in inf:
            key, val = line.split("\t")
            mapping[key] = val.strip()
    mapping[" "] = " "
    # language model
    scorer = Scorer(
        loader.lm_alpha,
        loader.lm_beta,
        loader.lm_binary_path,
        loader.lm_trie_path,
        alphabet,
    )
    # if you need to try greedy decoder without LM
    # decoded, log_prob = ctc_ops.ctc_greedy_decoder(logits, seq_length,
    #  merge_repeated=True)

    # Pad input to max_time_step of this batch
    source, source_lengths = pad_sequences(data)
    my_logits = sess.run(
        logits, feed_dict={input_tensor: source, seq_length: source_lengths}
    )
    my_logits = np.squeeze(my_logits)
    max_t, _ = my_logits.shape  # dim0=t, dim1=c

    # apply softmax
    res = np.zeros(my_logits.shape)
    for t in range(max_t):
        y = my_logits[t, :]
        e = np.exp(y)
        s = np.sum(e)
        res[t, :] = e / s

    decoded = ctc_beam_search_decoder(
        res,
        alphabet,
        loader.beam_width,
        cutoff_prob=loader.cutoff_prob,
        cutoff_top_n=loader.cutoff_top_n,
        scorer=scorer,
    )

    prediction = decoded[0][1].replace("\r", "")
    prediction = decodex(prediction, mapping)
    print(f"Results:{prediction}")
    return prediction


if __name__ == "__main__":
    data = [
        [705, 78.45454406738281, 0],
        [705.9090576171875, 78.45454406738281, 0],
        [706.8181762695312, 78.45454406738281, 0],
        [710.4545288085938, 77.54544830322266, 0],
        [712.272705078125, 76.63636016845703, 0],
        [719.54541015625, 73.90908813476562, 0],
        [721.3635864257812, 73.90908813476562, 0],
        [729.54541015625, 73, 0],
        [730.4545288085938, 73, 0],
        [735.9090576171875, 73.90908813476562, 0],
        [736.8181762695312, 73.90908813476562, 0],
        [737.7272338867188, 76.63636016845703, 0],
        [741.3635864257812, 77.54544830322266, 0],
        [745.9090576171875, 82.09090423583984, 0],
        [748.6363525390625, 83.90908813476562, 0],
        [754.0908813476562, 88.45454406738281, 0],
        [756.8181762695312, 90.27272033691406, 0],
        [757.7272338867188, 93, 0],
        [758.6363525390625, 93.90908813476562, 0],
        [758.6363525390625, 94.81817626953125, 0],
        [758.6363525390625, 95.7272720336914, 0],
        [758.6363525390625, 96.63636016845703, 0],
        [757.7272338867188, 96.63636016845703, 0],
        [755, 98.45454406738281, 0],
        [753.1817626953125, 98.45454406738281, 0],
        [752.272705078125, 98.45454406738281, 0],
        [749.54541015625, 99.36363220214844, 0],
        [747.7272338867188, 99.36363220214844, 0],
        [745, 99.36363220214844, 0],
        [742.272705078125, 99.36363220214844, 0],
        [740.4545288085938, 101.18181610107422, 0],
        [735, 101.18181610107422, 0],
        [734.0908813476562, 101.18181610107422, 0],
        [734.0908813476562, 100.27272033691406, 0],
        [731.3635864257812, 99.36363220214844, 0],
        [729.54541015625, 98.45454406738281, 0],
        [728.6363525390625, 98.45454406738281, 0],
        [728.6363525390625, 97.54544830322266, 0],
        [728.6363525390625, 96.63636016845703, 0],
        [728.6363525390625, 93.90908813476562, 0],
        [728.6363525390625, 93, 0],
        [728.6363525390625, 92.09090423583984, 0],
        [728.6363525390625, 91.18181610107422, 0],
        [728.6363525390625, 87.54544830322266, 0],
        [728.6363525390625, 86.63636016845703, 0],
        [728.6363525390625, 85.7272720336914, 0],
        [727.7272338867188, 83, 0],
        [727.7272338867188, 82.09090423583984, 0],
        [727.7272338867188, 81.18181610107422, 0],
        [726.8181762695312, 79.36363220214844, 0],
        [725.9090576171875, 79.36363220214844, 0],
        [725.9090576171875, 78.45454406738281, 0],
        [725.9090576171875, 77.54544830322266, 0],
        [725.9090576171875, 76.63636016845703, 0],
        [725.9090576171875, 75.7272720336914, 0],
        [726.8181762695312, 72.09090423583984, 0],
        [726.8181762695312, 71.18181610107422, 0],
        [726.8181762695312, 70.2727279663086, 0],
        [727.7272338867188, 69.36363220214844, 0],
        [727.7272338867188, 70.2727279663086, 0],
        [727.7272338867188, 72.09090423583984, 0],
        [729.54541015625, 73, 0],
        [731.3635864257812, 82.09090423583984, 0],
        [731.3635864257812, 85.7272720336914, 0],
        [734.0908813476562, 87.54544830322266, 0],
        [734.0908813476562, 93.90908813476562, 0],
        [734.0908813476562, 94.81817626953125, 0],
        [734.0908813476562, 95.7272720336914, 0],
        [727.7272338867188, 101.18181610107422, 0],
        [725.9090576171875, 102.09090423583984, 0],
        [725.9090576171875, 102.99999237060547, 0],
        [720.4545288085938, 105.7272720336914, 0],
        [720.4545288085938, 106.63636016845703, 0],
        [718.6363525390625, 106.63636016845703, 0],
        [717.7272338867188, 106.63636016845703, 0],
        [710.4545288085938, 105.7272720336914, 0],
        [708.6363525390625, 104.81817626953125, 0],
        [704.0908813476562, 102.99999237060547, 0],
        [703.1817626953125, 101.18181610107422, 0],
        [702.272705078125, 99.36363220214844, 0],
        [701.3635864257812, 99.36363220214844, 0],
        [701.3635864257812, 97.54544830322266, 0],
        [700.4545288085938, 97.54544830322266, 0],
        [700.4545288085938, 98.45454406738281, 0],
        [700.4545288085938, 99.36363220214844, 0],
        [700.4545288085938, 100.27272033691406, 0],
        [700.4545288085938, 104.81817626953125, 0],
        [699.54541015625, 106.63636016845703, 0],
        [697.7272338867188, 112.99999237060547, 0],
        [696.8181762695312, 115.7272720336914, 0],
        [694.0908813476562, 123, 0],
        [692.272705078125, 123, 0],
        [692.272705078125, 123.90908813476562, 0],
        [685, 128.4545440673828, 0],
        [678.6363525390625, 131.1818084716797, 0],
        [675.9090576171875, 133, 0],
        [670.4545288085938, 133.90908813476562, 0],
        [668.6363525390625, 133.90908813476562, 0],
        [668.6363525390625, 133, 0],
        [667.7272338867188, 132.0908966064453, 0],
        [666.8181762695312, 132.0908966064453, 0],
        [663.1818237304688, 128.4545440673828, 0],
        [663.1818237304688, 127.54545593261719, 0],
        [662.272705078125, 124.81817626953125, 0],
        [661.3635864257812, 119.36363220214844, 0],
        [661.3635864257812, 117.54544830322266, 0],
        [665.9090576171875, 105.7272720336914, 0],
        [667.7272338867188, 102.99999237060547, 0],
        [667.7272338867188, 101.18181610107422, 0],
        [683.1818237304688, 81.18181610107422, 0],
        [689.54541015625, 74.81817626953125, 0],
        [694.0908813476562, 70.2727279663086, 0],
        [699.54541015625, 66.63636016845703, 0],
        [700.4545288085938, 66.63636016845703, 0],
        [700.4545288085938, 65.7272720336914, 0],
        [701.3635864257812, 65.7272720336914, 1],
        [695.9090576171875, 99.36363220214844, 0],
        [694.0908813476562, 100.27272033691406, 0],
        [694.0908813476562, 101.18181610107422, 0],
        [694.0908813476562, 102.09090423583984, 1],
    ]
    prediction = do_inference(
        data,
        config_file="neural_network.ini",
        model_path="../models/model.ckpt-14",
        lm_binary_path="../models/lm/lm.binary",
        lm_trie_path="../models/lm/trie",
    )
