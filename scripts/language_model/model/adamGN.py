# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Weight updating functions."""
import warnings
import numpy
import mxnet as mx
from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import zeros, NDArray, full

__all__ = ['AdamwithGradientNormalization']

@register
class AdamwithGradientNormalization(Optimizer):
    """The Adam optimizer with weight decay regularization for BERT.

    Updates are applied by::

        rescaled_grad = clip(grad * rescale_grad, clip_gradient)
        m = beta1 * m + (1 - beta1) * rescaled_grad
        v = beta2 * v + (1 - beta2) * (rescaled_grad**2)
        w = w - learning_rate * (m / (sqrt(v) + epsilon) + wd * w)

    Note that this is different from `mxnet.optimizer.Adam`, where L2 loss is added and
    accumulated in m and v. In AdamW, the weight decay term decoupled from gradient
    based update.

    Parameters
    ----------
    beta1 : float, optional, default is 0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional, default is 0.999
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional, default is 1e-6
        Small value to avoid division by 0.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6,
                 **kwargs):
        super(AdamwithGradientNormalization, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def create_state_multi_precision(self, index, weight):
        """multi-precision state creation function."""
        weight_master_copy = None
        if self.multi_precision and weight.dtype == numpy.float16:
            weight_master_copy = weight.astype(numpy.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == numpy.float16 and not self.multi_precision:
            warnings.warn('Accumulating with float16 in optimizer can lead to '
                          'poor accuracy or slow convergence. '
                          'Consider using multi_precision=True option of the '
                          'BERTAdam optimizer')
        return self.create_state(index, weight)

    def create_state(self, _, weight):
        """state creation function."""
        return (zeros(weight.shape, weight.context, dtype=weight.dtype), #mean
                zeros(weight.shape, weight.context, dtype=weight.dtype)) #variance

    def update(self, index, weight, grad, state):
        """update function"""
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        """multi-precision update function"""
        use_multi_precision = self.multi_precision and weight.dtype == numpy.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)

    def _update_impl(self, indices, weight, grad, state, multi_precision=False):
        """update function"""
        try:
            from mxnet.ndarray.contrib import adamw_update
        except ImportError:
            raise ImportError('Failed to import nd.contrib.adamw_update from MXNet. '
                              'BERTAdam optimizer requires mxnet>=1.5.0b20190220. '
                              'Please upgrade your MXNet version. For example: '
                              'pip install mxnet-cu90 --pre. Otherwise, please consider '
                              'Adam optimizer with different hyper-parameters.')
        self._update_count(indices)
        lr = self._get_lr(indices)
        wd = self._get_wd(indices)
        t = self._index_update_count[indices]

        # pylint: disable=access-member-before-definition
        if not isinstance(self.rescale_grad, NDArray):
            self.rescale_grad = full(shape=(1,), val=self.rescale_grad, ctx=weight.context)
        else:
            self.rescale_grad = self.rescale_grad.as_in_context(weight.context)

        kwargs = {'beta1': self.beta1, 'beta2': self.beta2, 'epsilon': self.epsilon,
                  'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        grad = grad / mx.nd.norm(grad)
        if not multi_precision:
            mean, var = state
            adamw_update(weight, grad, mean, var, out=weight,
                         lr=1, wd=wd, eta=lr, **kwargs)
        else:
            try:
                from mxnet.ndarray.contrib import mp_adamw_update
            except ImportError:
                raise ImportError('Failed to import '
                                  'nd.contrib.mp_adamw_update from MXNet. '
                                  'BERTAdam optimizer requires mxnet>=1.5.0b20190220. '
                                  'Please upgrade your MXNet version. For example: '
                                  'pip install mxnet-cu90 --pre. Otherwise, please consider '
                                  'Adam optimizer with different hyper-parameters.')
            mean, var = state[0]
            mp_adamw_update(weight, grad, mean, var, state[1], out=weight,
                            lr=1, wd=wd, eta=lr, **kwargs)
