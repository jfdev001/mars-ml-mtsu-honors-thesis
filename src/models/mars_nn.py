"""Module for NNs for timeseries weather prediction.

Might use return_sequence and return_state args.
Remember, GRU returns are
Y = {h_t}_{t=1}^T
h_T
while LSTM returns are
Y = {h_t}_{t=1}^T
h_T
c_T

LSTM/GRU, Stacked GRU/LSTM, Conv-GRU/LSTM, AttnGRU/LSTM

Convolution and attention are probably too excessive for such
for medium range forecasting (3-7 days in advance).
Could also advance models for nowcasting since high resolution
spatio-temporal data is available. Imputation methods would
have to be considered for this methodology.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, \
    TimeDistributed, Bidirectional, \
    GRUCell, GRU, LSTMCell, LSTM, \
    Conv1D, MaxPool1D, BatchNormalization
from tensorflow.python.keras.layers.core import Dropout, Flatten
from tensorflow.python.ops.gen_batch_ops import Batch


class SimpleMarsNN(tf.keras.Model):
    """SimpleMarsNN is a recurrent neural net for timeseries forecasting.

    Options for LSTM vs. GRU and autoregressive vs single shot layers.
    """

    def __init__(self, output_tsteps, num_targets=1, autoregressive=False,
                 rnn_size=1, rnn_layers=1, rnn_cell='gru', dropout=0.0, **kwargs):
        """Define state for SimpleMarsRNN

        :param output_tsteps: <class 'int'>
        :param num_targets: <class 'int'>
        :param autoregressive: <class 'bool'> Determines whether the
            output is autoregressive or singleshot prediction.
        :param rnn_size: <class 'int'>
        :param rnn_layers: <class 'int'>
        :param rnn_cell: <class 'str'> Used for input RNN to extract
            embeddings from the input and/or used for the autoregressive
            output RNN if applicable.
        :param dropout: <class 'float'>
        """

        # Inheritance
        super().__init__(**kwargs)

        # Save args
        self.output_tsteps = output_tsteps
        self.num_targets = num_targets
        self.autoregressive = autoregressive

        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.rnn_cell = rnn_cell
        self.dropout = dropout

        # Validate
        if not((self.rnn_cell == 'gru') or (self.rnn_cell == 'lstm')):
            raise ValueError(':param rnn_cell: must be `gru` or `lstm`.')
        if self.rnn_layers <= 0:
            raise ValueError(':param rnn_layers: must be greater than 0.')

        # Define layers
        self.input_mars_rnn = MarsRNN(
            rnn_size=self.rnn_size,
            rnn_layers=self.rnn_layers,
            rnn_cell=self.rnn_cell,
            dropout=self.dropout,
            bidirectional=False)

        # Autoregresive output layers
        if self.autoregressive:
            self.output_ar_net = MarsARNet(
                rnn_size=self.rnn_size,
                rnn_cell=self.rnn_cell,
                tsteps=self.output_tsteps,
                dropout=self.dropout)

            # For stacking the output rnn after the autoregressive net.
            # the '- 1' ensures that the number of RNNs are symmetric...
            # i.e., if self.rnn_layers = 2 and the network is AR,
            # then 2 input rnns, 1 AR rnn, and then a followup output rnn
            # are used for a total of 4 RNN layers.
            if (self.rnn_layers - 1) > 0:
                self.output_mars_rnn = MarsRNN(
                    rnn_size=self.rnn_size,
                    rnn_layers=self.rnn_layers - 1,
                    rnn_cell=self.rnn_cell,
                    dropout=self.dropout,
                    bidirectional=False)

            self.td_dense = TimeDistributed(Dense(units=self.num_targets))

        # Singleshot output layers
        else:
            self.singleshot_dense = Dense(
                units=self.output_tsteps * self.num_targets)

            self.reshape = Reshape(target_shape=(
                self.output_tsteps, self.num_targets))

    def call(self, inputs):
        """Forward pass for SimpleMarsNN

        :param inputs: Rank-3 tensor (n, input_tsteps, features)

        :return: Rank-3 tensor (n, output_tsteps, num_targets)
        """

        # Determine layer call for input RNN
        if self.rnn_cell == 'gru':
            seqs, hidden_states = self.input_mars_rnn(inputs)

        else:
            seqs, hidden_states, cell_states = self.input_mars_rnn(inputs)

        # Autoregressive output
        if self.autoregressive:

            # Call the autoregressive neural net.
            if self.rnn_cell == 'gru':
                preds, ar_hidden_states = self.output_ar_net(
                    hidden_states, hidden_states)
            else:
                preds, ar_hidden_states, ar_cell_states = self.output_ar_net(
                    hidden_states, hidden_states, cell_states)

            # Determine stacked output rnn
            if (self.rnn_layers - 1) > 0:
                if self.rnn_cell == 'gru':
                    preds, output_rnn_hidden_states = self.output_mars_rnn(
                        preds, ar_hidden_states)
                else:
                    preds, output_rnn_hidden_states, output_rnn_cell_states = self.output_mars_rnn(
                        preds, ar_hidden_states, ar_cell_states)

            # Time distr outputs.
            # (batch, output_tsteps, rnn_units) => (batch, output_tsteps, num_targets)
            outputs = self.td_dense(preds)

        # Singleshot output
        else:

            # The last hidden states for either the lstm or the gru
            # are input to the dense layer. This follows the logic in
            # https://www.tensorflow.org/tutorials/structured_data/time_series#single-shot_models
            # (batch, rnn_size) => (batch, output_tsteps * num_targets)
            preds = self.singleshot_dense(hidden_states)

            # Reshape to sequence shape.
            # (batch, output_tsteps * num_targets) => (batch, output_tsteps, num_targets)
            outputs = self.reshape(preds)

        # Result of forward pass
        return outputs


class ConvMarsNN(tf.keras.Model):
    """Convolutional Mars Neural Net.

    Input(n, t_x, features)
        -> CNN ... batch norm and max pool
        -> Flat
        -> Dense(tsteps * features)
        -> Reshape
    """

    def __init__(self, output_tsteps,
                 activation, filters, kernel_size, conv_layers=1,
                 filter_increase_rate=1, conv_padding='valid', dilation_rate=1, conv_strides=1,
                 use_pooling=False, pool_size=2, pool_strides=None, pool_padding='valid',
                 dropout_rate=0.0,
                 num_targets=1, **kwargs):
        """Define state for ConvMarsNN.

        :param output_tsteps: <class 'int'>
        :param activation: <class 'str'> or <class 'tf.keras.activations'>
        :param filters: <class 'int'>
        :param kernel_size: <class 'int'> or <class 'tuple'> of <class 'int'>
        :param conv_layers: <class 'int'> Number of convolutional layers
            to include.
        :param filter_increase_rate: <class 'int'> The rate at which
            filters should increase as the number of layers increases.
            Subsequent layers will have a number of filters
            equal to `filters * filter_increase_rate`.
            E.g., conv_layer1(filters=16), and filter_increase_rate=2,
            then conv_layer2(filters=32)... so on and so forth.
        :param conv_padding: <class 'str'> in ['valid', 'same', 'causal']
        :param dilation_rate: <class 'int'>
        :param conv_strides: <class 'int'>
        :param use_pooling: <class 'bool'> True to use a pooling layer,
            False otherwise.
        :param pool_size: <class 'int'>
        :param pool_strides: <class 'int'>
        :param pool_padding: <class 'str'> in ['valid', 'same']
        :param dropout_rate: <class 'float'> in [0.0 --> 1.0]
        :param num_targets: <class 'int'>
        """

        # Inheritance
        super().__init__(**kwargs)

        # Validation
        if conv_layers <= 0:
            raise ValueError(':param conv_layers: must be > 0')

        # Save args for dense layer
        self.output_tsteps = output_tsteps
        self.num_targets = num_targets

        # Num layers for block
        self.conv_layers = conv_layers

        # Block args
        self.activation = activation
        self.filters = filters
        self.kernel_size = kernel_size
        self.filter_increase_rate = filter_increase_rate
        self.conv_padding = conv_padding
        self.dilation_rate = dilation_rate
        self.conv_strides = conv_strides

        self.use_pooling = use_pooling
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.pool_padding = pool_padding

        self.dropout_rate = dropout_rate

        # Define convolutional block
        self.conv_block = tf.keras.Sequential()
        for lyr in range(self.conv_layers):
            self.conv_block.add(CNNBlock(
                activation=self.activation,
                # Increase filters in subquent layers by rate...
                filters=self.filters * self.filter_increase_rate if lyr != 0 else self.filters,
                kernel_size=self.kernel_size,
                conv_padding=self.conv_padding,
                dilation_rate=self.dilation_rate**lyr,
                conv_strides=self.conv_strides,
                use_pooling=self.use_pooling,
                pool_size=self.pool_size,
                pool_strides=self.pool_strides,
                pool_padding=self.pool_padding,
                dropout_rate=self.dropout_rate))

        # Output of convolutional block will be rank-3 tensor
        self.flatten = Flatten()

        # Fully connected layer whose outputs will match timestep * dim
        self.dense = Dense(units=self.output_tsteps*self.num_targets)

        # Reshape fully connected output to rank-3 tensor (batch, time, dims)
        self.reshape = Reshape(
            target_shape=[self.output_tsteps, self.num_targets])

    def call(self, inputs):
        """Forward call for ConvMarsNN.

        :param inputs: Rank-3 tensor (batch, x_timesteps, x_dimensions)

        :return: Rank-3 tensor (batch, y_timesteps, y_dimensions)
        """

        conv_outputs = self.conv_block(inputs)
        flat_outputs = self.flatten(conv_outputs)
        dense_outputs = self.dense(flat_outputs)
        reshaped_outputs = self.reshape(dense_outputs)
        return reshaped_outputs


class CNNBlock(tf.keras.layers.Layer):
    """Generic Convolutional Block with BatchNorm and Max Pooling."""

    def __init__(
            self,
            activation, filters, kernel_size, conv_padding='valid', dilation_rate=1, conv_strides=1,
            use_pooling=True, pool_size=2, pool_strides=None, pool_padding='valid',
            dropout_rate=0.0,
            **kwargs):
        """Define state for CNNBlock.

        :param activation: <class 'str'> or <class 'tf.keras.activations'>
        :param filters: <class 'int'>
        :param kernel_size: <class 'int'> or <class 'tuple'> of <class 'int'>
        :param conv_padding: <class 'str'> in ['valid', 'same', 'causal']
        :param dilation_rate: <class 'int'>
        :param conv_strides: <class 'int'>
        :param use_pooling: <class 'bool'> True to use a pooling layer,
            False otherwise.
        :param pool_size: <class 'int'>
        :param pool_strides: <class 'int'>
        :param pool_padding: <class 'str'> in ['valid', 'same']
        :param dropout_rate: <class 'float'> in [0.0 --> 1.0]
        """

        # Inheritance
        super().__init__(**kwargs)

        # Conv1d args
        self.activation = activation
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_padding = conv_padding
        self.dilation_rate = dilation_rate
        self.conv_strides = conv_strides

        # Pool args
        self.use_pooling = use_pooling
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.pool_padding = pool_padding

        # Dropout
        self.dropout_rate = dropout_rate

        # Define layers
        self.conv_layer = Conv1D(
            filters=self.filters, kernel_size=self.kernel_size,
            padding=self.conv_padding,
            activation=self.activation,
            dilation_rate=self.dilation_rate)

        if self.use_pooling:
            self.max_pool = MaxPool1D(
                pool_size=self.pool_size,
                strides=self.pool_strides,
                padding=self.pool_padding)

        self.batch_norm = BatchNormalization()

        self.dropout = Dropout(rate=self.dropout_rate)

    def call(self, inputs):
        """Forward pass for CNNBlock.

        :param inputs: Rank-3 tensor of (batch_size, timesteps, dimensions)

        :return: Rank-3 convolved tensor
        """

        conv_outputs = self.conv_layer(inputs)

        if self.use_pooling:
            conv_outputs = self.max_pool(conv_outputs)

        norm_outputs = self.batch_norm(conv_outputs)
        dropped_outputs = self.dropout(norm_outputs)
        return dropped_outputs


class MarsTransformer(tf.keras.Model):
    """Multihead attention Mars Neural Net"""

    def __init__(self, **kwargs):
        pass

    def call(self, inputs):
        pass


class MarsRNN(tf.keras.layers.Layer):
    """Mars (opt bidirectional) recurrent neural net.

    TODO: Make the RNN stack more readable with sequential layer
    and stateful kwarg?
    """

    def __init__(self, rnn_size, rnn_layers, rnn_cell, dropout, bidirectional, **kwargs):
        """Define state for MarsNN

        :param rnn_size: <class 'int'>
        :param rnn_layers: <class 'int'>
        :param rnn_cell: <class 'str'>
        """

        # Inheritance
        super().__init__(**kwargs)

        # Save args
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.rnn_cell = rnn_cell
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Define GRU/LSTM layers -- have to explicitly use tf.keras.layers
        # to avoid pass by ref issue
        if self.rnn_cell == 'gru':

            # List of layers of bidirectional (if applicable) wrapped GRU
            self.rnn = [
                Bidirectional(GRU(
                    units=self.rnn_size,
                    return_sequences=True,
                    return_state=True,
                    dropout=self.dropout))
                if self.bidirectional
                else GRU(
                    units=self.rnn_size,
                    return_sequences=True,
                    return_state=True,
                    dropout=self.dropout)
                for i in range(self.rnn_layers)]

        elif self.rnn_cell == 'lstm':

            # List of layers of bidirectional (if applicable) wrapped LSTM
            self.rnn = [
                Bidirectional(LSTM(
                    units=self.rnn_size,
                    return_sequences=True,
                    return_state=True,
                    dropout=self.dropout))
                if self.bidirectional
                else LSTM(
                    units=self.rnn_size,
                    return_sequences=True,
                    return_state=True,
                    dropout=self.dropout)
                for i in range(self.rnn_layers)]
        else:
            raise ValueError(
                '`:param rnn_layer_type:` must be `gru` or `lstm`')

    def call(self, seqs, hidden_states=None, cell_states=None):
        """Forward pass for stacked RNN

        TODO: Should zeros be in call or attached to obj state?

        :param seqs: Rank-3 tensor (n, t, features)
        :param hidden_states: <class 'list'> of tensors if bidirectional,
            single tensor otherwise.
        :param cell_states: <class 'list'> of tensors if bidirectional,
            single tensor otherwise

        :return: <class 'tuple'>
            (1) Rank-3 tensor of all hidden states (n, t, rnn_units)
            (2) Rank-2 tensor for last hidden state (n, rnn_units)
            ? (3) Rank-2 tensor for last cell state (n, rnn_units)
        """

        # Default zeros tensor
        if (hidden_states is None) or (cell_states is None):
            # print(seqs)
            # print(seqs.shape)
            # breakpoint()
            zeros = tf.zeros(shape=(seqs.shape[0], self.rnn_size))

        # Assign unidirectional hidden states
        if hidden_states is None:
            fwd_hidden_states = zeros
        elif not isinstance(hidden_states, list):
            fwd_hidden_states = hidden_states

        # Assigning unidirectional cell states
        if (self.rnn_cell == 'lstm') and (cell_states is None):
            fwd_cell_states = zeros
        elif (self.rnn_cell == 'lstm') and (not isinstance(cell_states, list)):
            fwd_cell_states = cell_states

        # Default bidirectional hidden state
        if (self.bidirectional) and (hidden_states is None):
            bckwd_hidden_states = zeros

        # Default bidirectional cell state
        if (self.bidirectional) and (cell_states is None):
            bckwd_cell_states = zeros

        # Extracting multiple hidden states from args if applicable
        if isinstance(hidden_states, list):
            fwd_hidden_states, bckwd_hidden_states = hidden_states

        # Extracting multiple cell states if applicable
        if isinstance(cell_states, list):
            fwd_cell_states, bckwd_cell_states = cell_states

        # Forward call through layers
        for layer in self.rnn:

            # Determine cell
            if self.rnn_cell == 'gru':

                # Determine bidirectionality
                if self.bidirectional:
                    seqs, fwd_hidden_states, bckwd_hidden_states = layer(
                        seqs, initial_state=[fwd_hidden_states, bckwd_hidden_states])
                else:
                    seqs, fwd_hidden_states = layer(
                        seqs, initial_state=fwd_hidden_states)

            else:

                # Determine bidirectionality
                if self.bidirectional:
                    seqs,
                    fwd_hidden_states, bckwd_hidden_states
                    fwd_cell_states, bckwd_cell_states = layer(
                        seqs, initial_state=[
                            fwd_hidden_states, fwd_cell_states,
                            bckwd_hidden_states, bckwd_cell_states])
                else:
                    seqs, fwd_hidden_states, fwd_cell_states = layer(
                        seqs, initial_state=[fwd_hidden_states, fwd_cell_states])

        # Result of forward call
        if self.rnn_cell == 'gru':
            if self.bidirectional:
                return seqs, fwd_hidden_states, bckwd_hidden_states
            else:
                return seqs, fwd_hidden_states

        else:
            if self.bidirectional:
                return seqs, \
                    fwd_hidden_states, fwd_cell_states, \
                    bckwd_hidden_states, bckwd_cell_states
            else:
                return seqs, fwd_hidden_states, fwd_cell_states


class MarsARNet(tf.keras.Model):
    """Mars Autoregressive (AR) Net."""

    def __init__(self, rnn_size, rnn_cell, tsteps, dropout, **kwargs):
        """Defines state for MarsARNet.

        :param rnn_size: <class 'int'>
        :param rnn_cell: <class 'str'>
        :param tsteps: <class 'int'>
        :param dropout: <class 'float'>
        """

        # Inheritance
        super().__init__(**kwargs)

        # Save args
        self.rnn_size = rnn_size
        self.rnn_cell = rnn_cell
        self.tsteps = tsteps
        self.dropout = dropout

        # Validate
        if not ((self.rnn_cell == 'gru') or (self.rnn_cell == 'lstm')):
            raise ValueError(':param rnn_cell: must be `gru` or `lstm`.')

        # Define cell
        if self.rnn_cell == 'gru':
            self.cell = GRUCell(units=self.rnn_size)
        else:
            self.cell = LSTMCell(units=self.rnn_size)

    def call(self, cell_inputs, hidden_states=None, cell_states=None):
        """Forward pass for MarsARNet.

        Uses the hidden states (h_T) of an initial
        RNN (the 'warmup' in https://www.tensorflow.org/tutorials/structured_data/time_series#advanced_autoregressive_model)
        to inform the initial cell input and hidden state,
        after which the cell will unroll until the desired output
        time steps counter is reached.
        """

        # Default states
        if (hidden_states is None) or (cell_states is None):
            zeros = tf.zeros(shape=(cell_inputs.shape[0], self.rnn_size))
            hidden_states = zeros
            cell_states = zeros

        # Track the sequential output of the rnn
        outputs = tf.TensorArray(dtype=tf.float32, size=self.tsteps)

        # Autoregressive update of `cell_input` using rnn cell
        for t in tf.range(self.tsteps):
            if self.rnn_cell == 'gru':
                # (batch, units) => (batch, units) & (batch, units)
                cell_inputs, hidden_states = self.cell(
                    inputs=cell_inputs, states=hidden_states)
            else:
                # (batch, units) => (batch, units), (batch, units), & (batch, units)
                cell_inputs, hidden_and_cell_states = self.cell(
                    inputs=cell_inputs, states=[hidden_states, cell_states])

                # Extract hidden and cell states
                hidden_states, cell_states = hidden_and_cell_states

            # Write to tensor array
            outputs = outputs.write(index=t, value=cell_inputs)

        # Transpose the outputs tensor from (timesteps, n, units) => (n, timesteps, units)
        outputs = tf.transpose(outputs.stack(), perm=[1, 0, 2])

        # Result of forward pass
        if self.rnn_cell == 'gru':
            return outputs, hidden_states
        else:
            return outputs, hidden_states, cell_states
