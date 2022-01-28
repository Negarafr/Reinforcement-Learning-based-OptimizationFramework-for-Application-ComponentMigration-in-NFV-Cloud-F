
import numpy as np
import tensorflow as tf
def vector_embedding(inputBatch):
    state_size_sequence = inputBatch.maxServiceLength
    state_size_embeddings = inputBatch.num_vnfs #number of node
    state = np.zeros((inputBatch.request, state_size_sequence, state_size_embeddings), dtype='int32')
    for batch in range(inputBatch.request):
        for i in range(inputBatch.serviceLength[batch]):
            embedding = inputBatch.state[batch][i]
            state[batch][i][embedding] = 1
    return state
class DynamicMultiRNN(object):
        #Implementation of a dynamic multi-cell RNN
    def __init__(self, action_size, batch_size, input_, input_len_, num_activations, num_layers):
        self.action_size = action_size
        self.batch_size = batch_size
        self.num_activations = num_activations
        self.num_layers = num_layers
        self.positions = []
        self.outputs = []
        self.input_ = input_
        self.input_len_ = input_len_
        # Variables initializer
        initializer = tf.contrib.layers.xavier_initializer()
        # Generate multiple LSTM cell
        cells = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(self.num_activations, state_is_tuple=True) for _ in range(self.num_layers)], state_is_tuple=True)
        # LSTMs internal state
        c_initial_states = []
        h_initial_states = []
        # Initial state (tuple) is trainable but same for all batch
        for i in range(self.num_layers):
            first_state = tf.get_variable("var{}".format(i), [1, self.num_activations], initializer=initializer)
            first_state = tf.Print(first_state, ["first_state", first_state], summarize=10)
            c_initial_state = tf.tile(first_state, [self.batch_size, 1])
            h_initial_state = tf.tile(first_state, [self.batch_size, 1])
            c_initial_states.append(c_initial_state)
            h_initial_states.append(h_initial_state)
        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(c_initial_states[idx], h_initial_states[idx])
             for idx in range(self.num_layers)]
        )

        states_series, current_state = tf.nn.dynamic_rnn(cells, input_, initial_state=rnn_tuple_state, sequence_length=input_len_)
        self.outputs = tf.layers.dense(states_series, self.action_size, activation=tf.nn.softmax)       # [Batch, seq_length, action_size]
        self.outputs = tf.Print(self.outputs, ["outputs", self.outputs, tf.shape(self.outputs)],summarize=10)

        # Multinomial distribution
        prob = tf.contrib.distributions.Categorical(probs=self.outputs)
        # Sample from distribution
        self.positions = prob.sample()        # [Batch, seq_length]
        self.positions = tf.cast(self.positions, tf.int32)
        self.positions = tf.Print(self.positions, ["position", self.positions, tf.shape(self.positions)], summarize=10)
class Agent:
    def __init__(self, state_size_embeddings, state_maxServiceLength, action_size, batch_size, learning_rate, hidden_dim,  num_stacks):
        # Training config (agent)
        self.learning_rate = learning_rate
        self.action_size = action_size
        self.batch_size = batch_size
        self.state_size_embeddings = state_size_embeddings
        self.state_maxServiceLength = state_maxServiceLength
        self.hidden_dim = hidden_dim
        self.num_stacks = num_stacks
        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = tf.placeholder(tf.float32, [self.batch_size, self.state_maxServiceLength, self.state_size_embeddings], name="input")
        self.input_len_ = tf.placeholder(tf.float32, [self.batch_size], name="input_len")
        self._build_model()
        self._build_optimization()
        self.merged = tf.summary.merge_all()

    def _build_model(self):
        with tf.variable_scope('multi_lstm'):
            self.ptr = DynamicMultiRNN(self.action_size, self.batch_size, self.input_, self.input_len_, self.hidden_dim, self.num_stacks)
    def _build_optimization(self):

        with tf.name_scope('reinforce'):

            self.reward_holder = tf.placeholder(tf.float32, [self.batch_size], name="reward_holder")
            self.positions_holder = tf.placeholder(tf.float32, [self.batch_size, self.state_maxServiceLength], name="positions_holder")
            # Optimizer learning rate
            #self.opt = tf.train.exponential_decay(self.lr1_start, self.global_step, self.lr1_decay_step, self.lr1_decay_rate, staircase=False, name="learning_rate1")
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99, epsilon=0.0000001)

            # Multinomial distribution
            probs = tf.contrib.distributions.Categorical(probs=self.ptr.outputs)
            log_softmax = probs.log_prob(self.positions_holder)         # [Batch, seq_length]
            #log_softmax = tf.Print(log_softmax, ["log_softmax", log_softmax, tf.shape(log_softmax)])


            log_softmax_mean = tf.reduce_sum(log_softmax,1)

            reward = tf.divide(1000.0, self.reward_holder, name="div")
            reward = tf.Print(reward, ["reward", reward])

            reward = tf.stop_gradient(reward)

            # Compute Loss
            loss = tf.reduce_mean(reward * log_softmax_mean, 0)     # Scalar
            loss = tf.Print(loss, ["loss", loss])
            tf.summary.scalar('loss', loss)

            # Minimize step
            gvs = opt.compute_gradients(loss)

            #Clipping
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None]  # L2 clip

            self.train_step = opt.apply_gradients(capped_gvs)


if __name__ == "__main__":

    pass
