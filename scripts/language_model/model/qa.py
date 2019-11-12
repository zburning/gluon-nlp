
"""XLNetForQA models."""


import mxnet as mx
from mxnet.gluon import HybridBlock, Block, loss, nn
from mxnet.gluon.loss import Loss

class XLNetPoolerAnswerClass(HybridBlock):
    """ Compute SQuAD 2.0 answer class from classification and start tokens hidden states. """
    def __init__(self, units = 768, prefix=None, params=None):
        super(XLNetPoolerAnswerClass, self).__init__(prefix=prefix, params=params)
        self._units = units
        self.dense_0 = nn.Dense(units, activation='tanh', prefix=prefix)
        self.dense_1 = nn.Dense(1, use_bias=False)

    def __call__(self, hidden_states, start_positions=None, start_states=None, cls_index=None):
        return super(XLNetPoolerAnswerClass, self).__call__(hidden_states, start_positions, cls_index)

    def hybrid_forward(self, F, hidden_states, start_positions=None, cls_index=None):
        #get the cls_token's state, currently the last state
        cls_token_state = hidden_states.slice(begin=(0, -1, 0), end=(None, -2, None), step=(None, -1, None))
        cls_token_state = cls_token_state.reshape(shape=(-1, self._units))
        if start_positions is not None:
            start_states = []
            for (i,seq) in enumerate(hidden_states):
                start_state = F.take(seq, start_positions[i], axis=0).squeeze(0)
                start_states.append(start_state)
            start_states = mx.ndarray.stack(*start_states, axis=0)
        if start_state is not None:
            x = self.dense_0(mx.ndarray.concat(start_states, cls_token_state, dim=-1))
        else:
            x = self.dense_0(cls_token_state)
        x = self.dense_1(x).squeeze(-1)
        return x

class XLNetForQA(Block):
    """Model for SQuAD task with XLNet.

    Parameters
    ----------
    bert: XLNet base
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    """

    def __init__(self, xlnet_base, prefix=None, params=None):
        super(XLNetForQA, self).__init__(prefix=prefix, params=params)
        self.xlnet = xlnet_base
        with self.name_scope():
            self.span_classifier = nn.Dense(units=2, flatten=False)

    def __call__(self, inputs, token_types, valid_length=None):
        #pylint: disable=arguments-differ, dangerous-default-value
        """Generate the unnormalized score for the given the input sequences."""
        # XXX Temporary hack for hybridization as hybridblock does not support None inputs
        valid_length = [] if valid_length is None else valid_length
        return super(XLNetForQA, self).__call__(inputs, token_types, valid_length)

    def _padding_mask(self, inputs, valid_length_start, left_pad=True):
        F = mx.ndarray
        if left_pad:
        #left pad
            valid_length_start = valid_length_start
            steps = F.contrib.arange_like(inputs, axis=1) - 1
            ones = F.ones_like(steps)
            mask = F.broadcast_greater(F.reshape(steps, shape=(1, -1)),
                                       F.reshape(valid_length_start, shape=(-1, 1)))
            mask = F.broadcast_mul(F.expand_dims(mask, axis=1),
                                   F.broadcast_mul(ones, F.reshape(ones, shape=(-1, 1))))
        else:
            raise NotImplementedError
        return mask

    def forward(self, inputs, token_types, valid_length=None, mems=None):
        # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray, shape (batch_size, seq_length)
            Input words for the sequences.
        token_types : NDArray, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length : NDArray or None, shape (batch_size,)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, seq_length, 2)
        """
        # XXX Temporary hack for hybridization as hybridblock does not support None inputs
        if isinstance(valid_length, list) and len(valid_length) == 0:
            valid_length = None
        valid_length_start = inputs.shape[1] - valid_length
        attention_mask = self._padding_mask(inputs, valid_length_start).astype('float32')
        output, _ = self.xlnet(inputs, token_types, mems, attention_mask)
        span_output = self.span_classifier(output)
        return (output, span_output)


class XLNetForQALoss(Loss):
    """Loss for SQuAD task with XLNet.

    """

    def __init__(self, weight=None, batch_axis=0, **kwargs):  # pylint: disable=unused-argument
        super(XLNetForQALoss, self).__init__(
            weight=None, batch_axis=0, **kwargs)
        self.loss = loss.SoftmaxCELoss()
        self.cls_loss = loss.SigmoidBinaryCrossEntropyLoss()
        self.answerpooling = XLNetPoolerAnswerClass()

    def __call__(self, pred, hidden_states, label, is_impossible=None):
        return super(XLNetForQALoss, self).__call__(pred, hidden_states, label, is_impossible)

    def hybrid_forward(self, F, pred, hidden_states, label, is_impossible = None):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        pred : NDArray, shape (batch_size, seq_length, 2)
            XLNetSquad forward output.
        hidden_states: NDArray, shape (batch_size, seq_length, units)
        label : list, length is 2, each shape is (batch_size,1)
            label[0] is the starting position of the answer,
            label[1] is the ending position of the answer.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size,)
        """
        #span loss

        pred = F.split(pred, axis=2, num_outputs=2)
        start_pred = pred[0].reshape((0, -3))
        start_label = label[0]
        end_pred = pred[1].reshape((0, -3))
        end_label = label[1]
        span_loss = (self.loss(start_pred, start_label) + self.loss(
            end_pred, end_label)) / 2
        #regression loss
        cls_loss = None
        if is_impossible is not None:
            cls_logits = self.answerpooling(hidden_states, start_positions=start_label)
            cls_loss = self.cls_loss(cls_logits, is_impossible)
        total_loss = span_loss + cls_loss if cls_loss is not None else span_loss
        return total_loss
