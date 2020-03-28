import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Layer, RNN, Reshape
import tensorflow.keras.backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.layers.recurrent import _generate_dropout_mask, _generate_zero_filled_state_for_cell
from tensorflow.python.keras.engine.input_spec import InputSpec


class JujubeCakeCell(Layer):
    def __init__(self,
                 sub_units,
                 sub_lstms,

                 cake_activation='tanh',
                 cake_recurrent_activation='hard_sigmoid',
                 sub_activation='tanh',
                 sub_recurrent_activation='hard_sigmoid',

                 cake_use_bias=True,
                 sub_use_bias=True,

                 cake_kernel_initializer='glorot_uniform',
                 cake_recurrent_initializer='orthogonal',
                 cake_bias_initializer='zeros',
                 sub_kernel_initializer='glorot_uniform',
                 sub_recurrent_initializer='orthogonal',
                 sub_bias_initializer='zeros',

                 cake_unit_forget_bias=True,
                 sub_unit_forget_bias=True,

                 cake_kernel_regularizer=None,
                 cake_recurrent_regularizer=None,
                 cake_bias_regularizer=None,
                 sub_kernel_regularizer=None,
                 sub_recurrent_regularizer=None,
                 sub_bias_regularizer=None,

                 cake_kernel_constraint=None,
                 cake_recurrent_constraint=None,
                 cake_bias_constraint=None,
                 sub_kernel_constraint=None,
                 sub_recurrent_constraint=None,
                 sub_bias_constraint=None,

                 cake_dropout=0.,
                 cake_recurrent_dropout=0.,
                 sub_dropout=0.,
                 sub_recurrent_dropout=0.,

                 implementation=1,

                 **kwargs):
        super(JujubeCakeCell, self).__init__(**kwargs)
        self.sub_units = sub_units
        self.sub_lstms = sub_lstms
        self.units = self.sub_units * self.sub_lstms

        self.cake_activation = activations.get(cake_activation)
        self.cake_recurrent_activation = activations.get(
            cake_recurrent_activation)
        self.sub_activation = activations.get(sub_activation)
        self.sub_recurrent_activation = activations.get(
            sub_recurrent_activation)
        self.cake_use_bias = cake_use_bias
        self.sub_use_bias = sub_use_bias

        self.cake_kernel_initializer = initializers.get(
            cake_kernel_initializer)
        self.cake_recurrent_initializer = initializers.get(
            cake_recurrent_initializer)
        self.cake_bias_initializer = initializers.get(cake_bias_initializer)
        self.sub_kernel_initializer = initializers.get(sub_kernel_initializer)
        self.sub_recurrent_initializer = initializers.get(
            sub_recurrent_initializer)
        self.sub_bias_initializer = initializers.get(sub_bias_initializer)

        self.cake_unit_forget_bias = cake_unit_forget_bias
        self.sub_unit_forget_bias = sub_unit_forget_bias

        self.cake_kernel_regularizer = regularizers.get(
            cake_kernel_regularizer)
        self.cake_recurrent_regularizer = regularizers.get(
            cake_recurrent_regularizer)
        self.cake_bias_regularizer = regularizers.get(cake_bias_regularizer)
        self.sub_kernel_regularizer = regularizers.get(sub_kernel_regularizer)
        self.sub_recurrent_regularizer = regularizers.get(
            sub_recurrent_regularizer)
        self.sub_bias_regularizer = regularizers.get(sub_bias_regularizer)

        self.cake_kernel_constraint = constraints.get(cake_kernel_constraint)
        self.cake_recurrent_constraint = constraints.get(
            cake_recurrent_constraint)
        self.cake_bias_constraint = constraints.get(cake_bias_constraint)
        self.sub_kernel_constraint = constraints.get(sub_kernel_constraint)
        self.sub_recurrent_constraint = constraints.get(
            sub_recurrent_constraint)
        self.sub_bias_constraint = constraints.get(sub_bias_constraint)

        self.cake_dropout = min(1., max(0., cake_dropout))
        self.cake_recurrent_dropout = min(1., max(0., cake_recurrent_dropout))
        self.sub_dropout = min(1., max(0., sub_dropout))
        self.sub_recurrent_dropout = min(1., max(0., sub_recurrent_dropout))

        self.implementation = implementation

        self.state_size = [self.units,
                           self.units,
                           self.sub_units,
                           self.sub_units]
        self.sub_state_size = [self.sub_units, self.sub_units]

        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        self._sub_dropout_mask = None
        self._sub_recurrent_dropout_mask = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        # print('input_dim:', input_dim)
        sub_input_dim = int(int(input_dim) / int(self.sub_lstms))
        # print('sub_input_dim:', sub_input_dim)
        self.cake_kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name='cake_kernel',
            initializer=self.cake_kernel_initializer,
            regularizer=self.cake_kernel_regularizer,
            constraint=self.cake_kernel_constraint)
        # print('cake_kernel.shape:', self.cake_kernel.shape)
        self.cake_recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='cake_recurrent_kernel',
            initializer=self.cake_recurrent_initializer,
            regularizer=self.cake_recurrent_regularizer,
            constraint=self.cake_recurrent_constraint)
        # print('cake_recurrent_kernel.shape', self.cake_recurrent_kernel.shape)
        self.sub_kernel = self.add_weight(
            shape=(sub_input_dim, self.sub_units * 4),
            name='sub_kernel',
            initializer=self.sub_kernel_initializer,
            regularizer=self.sub_kernel_regularizer,
            constraint=self.sub_kernel_constraint)
        self.sub_recurrent_kernel = self.add_weight(
            shape=(self.sub_units, self.sub_units * 4),
            name='sub_recurrent_kernel',
            initializer=self.sub_recurrent_initializer,
            regularizer=self.sub_recurrent_regularizer,
            constraint=self.sub_recurrent_constraint)

        if self.cake_use_bias:
            if self.cake_unit_forget_bias:

                def cake_bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.cake_bias_initializer(
                            (self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.cake_bias_initializer(
                            (self.units,), *args, **kwargs),
                    ])
            else:
                cake_bias_initializer = self.cake_bias_initializer
            self.cake_bias = self.add_weight(
                shape=(self.units * 3,),
                name='cake_bias',
                initializer=cake_bias_initializer,
                regularizer=self.cake_bias_regularizer,
                constraint=self.cake_bias_constraint)
        else:
            self.cake_bias = None

        if self.sub_use_bias:
            if self.sub_unit_forget_bias:

                def sub_bias_initializer(_, *args, **kwagrs):
                    return K.concatenate([
                        self.sub_bias_initializer(
                            (self.sub_units,), *args, **kwagrs),
                        initializers.Ones()((self.sub_units,), *args, **kwagrs),
                        self.sub_bias_initializer(
                            (self.sub_units * 2,), *args, **kwagrs),
                    ])
            else:
                sub_bias_initializer = self.sub_bias_initializer
            self.sub_bias = self.add_weight(
                shape=(self.sub_units * 4,),
                name='sub_bias',
                initializer=sub_bias_initializer,
                regularizer=self.sub_bias_regularizer,
                constraint=self.sub_bias_constraint)
        else:
            self.sub_bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        sub_states = states[2:]
        states = states[:2]

        h_tm1 = states[0]
        c_tm1 = states[1]

        sub_h_tm1 = sub_states[0]
        sub_c_tm1 = sub_states[1]
        if 0 < self.cake_dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                array_ops.ones_like(inputs),
                self.cake_dropout,
                training=training,
                count=3)
        if (0 < self.cake_recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                array_ops.ones_like(states[0]),
                self.cake_recurrent_dropout,
                training=training,
                count=3)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        sub_inputs = tf.split(inputs, self.sub_lstms, -1)
        c_new = []
        for item in sub_inputs:
            sub_h_tm1, [sub_h_tm1, sub_c_tm1] = self._subcall(
                item, [sub_h_tm1, sub_c_tm1], training)
            c_new.append(sub_c_tm1)
        sub_h = sub_h_tm1
        sub_c = sub_c_tm1
        c_new = K.concatenate(c_new)

        if self.implementation == 1:
            if 0 < self.cake_dropout < 1:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_o = inputs * dp_mask[2]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_o = inputs
            x_i = K.dot(inputs_i, self.cake_kernel[:, :self.units])
            x_f = K.dot(
                inputs_f, self.cake_kernel[:, self.units:self.units * 2])
            x_o = K.dot(
                inputs_o, self.cake_kernel[:, self.units * 2:self.units * 3])
            if self.cake_use_bias:
                x_i = K.bias_add(x_i, self.cake_bias[:self.units])
                x_f = K.bias_add(
                    x_f, self.cake_bias[self.units:self.units * 2])
                x_o = K.bias_add(x_o, self.cake_bias[self.units * 2:])

            if 0 < self.cake_recurrent_dropout < 1:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_o = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_o = h_tm1
            x = (x_i, x_f, x_o)
            h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_o)
            c, o = self._cake_compute_carry_and_output(x, h_tm1, c_tm1, c_new)
        else:
            if 0 < self.cake_dropout < 1:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.cake_kernel)
            if 0 < self.cake_recurrent_dropout < 1:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_tml, self.cake_recurrent_kernel)
            if self.cake_use_bias:
                z = K.bias_add(z, self.cake_bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units:self.units * 2]
            z1 = z[:, self.units * 2:]

            z = (z0, z1, z2)
            c, o = self._cake_compute_carry_and_output_fused(z, c_tm1, new_c)
        # print('c.shape:', c.shape)
        # print('o.shape:', o.shape)
        h = o * self.cake_activation(c)
        return h, [h, c, sub_h, sub_c]

    def _cake_compute_carry_and_output(self, x, h_tm1, c_tm1, c_new):
        """Computes carry and output using split kernels for cake cell."""
        x_i, x_f, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_o = h_tm1
        i = self.cake_recurrent_activation(
            x_i + K.dot(h_tm1_i, self.cake_recurrent_kernel[:, :self.units]))
        f = self.cake_recurrent_activation(x_f + K.dot(
            h_tm1_f, self.cake_recurrent_kernel[:, self.units:self.units * 2]))
        # print('i.shape:', i.shape)
        # print('c_new.shape:', c_new.shape)
        # print('f.shape:', f.shape)
        # print('c_tm1.shape:', c_tm1.shape)
        c = f * c_tm1 + i * c_new
        o = self.cake_recurrent_activation(
            x_o + K.dot(h_tm1_o, self.cake_recurrent_kernel[:, self.units * 2:]))
        return c, o

    def _cake_computs_carry_and_output_fused(self, x, c_tm1, c_new):
        """Computes carry and output using fused kernels for cake cell."""
        z0, z1, z2 = z
        i = self.cake_recurrent_activation(z0)
        f = self.cake_recurrent_activation(z1)
        c = f * c_tm1 + i * new_c
        o = self.cake_recurrent_activation(z2)
        return c, o

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Computes carry and output using split kernels for sub cell."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.sub_recurrent_activation(
            x_i + K.dot(h_tm1_i, self.sub_recurrent_kernel[:, :self.sub_units]))
        f = self.sub_recurrent_activation(x_f + K.dot(
            h_tm1_f, self.sub_recurrent_kernel[:, self.sub_units:self.sub_units * 2]))
        c = f * c_tm1 + i * self.sub_activation(x_c + K.dot(
            h_tm1_c, self.sub_recurrent_kernel[:, self.sub_units * 2:self.sub_units * 3]))
        o = self.sub_recurrent_activation(
            x_o + K.dot(h_tm1_o, self.sub_recurrent_kernel[:, self.sub_units * 3:]))
        return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels for sub cell."""
        z0, z1, z2, z3 = z
        i = self.sub_recurrent_activation(z0)
        f = self.sub_recurrent_activation(z1)
        c = f * c_tm1 + i * self.sub_activation(z2)
        o = self.sub_recurrent_activation(z3)
        return c, o

    def _subcall(self, inputs, states, training=None):
        """
        In this part, we imitate the implementation of Keras LSTMCell.
        """

        if 0 < self.sub_dropout < 1 and self._sub_dropout_mask is None:
            self._sub_dropout_mask = _generate_dropout_mask(
                array_ops.ones_like(inputs),
                self.sub_dropout,
                training=training,
                count=4)
        if (0 < self.sub_recurrent_dropout < 1 and
                self._sub_recurrent_dropout_mask is None):
            self._sub_recurrent_dropout_mask = _generate_dropout_mask(
                array_ops.ones_like(states[0]),
                self.sub_recurrent_dropout,
                training=training,
                count=4)
        dp_mask = self._sub_dropout_mask
        rec_dp_mask = self._sub_recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        # print('sub_inputs.shape:', inputs.shape)
        # print('sub_h_tm1.shape:', h_tm1.shape)
        # print('sub_c_tm1.shape:', c_tm1.shape)
        # print('sub_dp_mask.shape:', dp_mask[0].shape)
        # print('sub_rec_dp_mask.shape:', rec_dp_mask[0].shape)
        if self.implementation == 1:
            if 0 < self.sub_dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            x_i = K.dot(inputs_i, self.sub_kernel[:, :self.sub_units])
            x_f = K.dot(
                inputs_f, self.sub_kernel[:, self.sub_units:self.sub_units * 2])
            x_c = K.dot(
                inputs_c, self.sub_kernel[:, self.sub_units * 2:self.sub_units * 3])
            x_o = K.dot(inputs_o, self.sub_kernel[:, self.sub_units * 3:])
            if self.sub_use_bias:
                x_i = K.bias_add(x_i, self.sub_bias[:self.sub_units])
                x_f = K.bias_add(
                    x_f, self.sub_bias[self.sub_units:self.sub_units * 2])
                x_c = K.bias_add(
                    x_c, self.sub_bias[self.sub_units * 2:self.sub_units * 3])
                x_o = K.bias_add(x_o, self.sub_bias[self.sub_units * 3:])

            if 0 < self.sub_recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            x = (x_i, x_f, x_c, x_o)
            h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
            c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
        else:
            if 0. < self.sub_dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.sub_kernel)
            if 0. < self.sub_recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_tm1, self.sub_recurrent_kernel)
            if self.sub_use_bias:
                z = K.bias_add(z, self.sub_bias)

            z0 = z[:, :self.sub_units]
            z1 = z[:, self.sub_units:2 * self.sub_units]
            z2 = z[:, 2 * self.sub_units:3 * self.sub_units]
            z3 = z[:, 3 * self.sub_units:]

            z = (z0, z1, z2, z3)
            c, o = self._compute_carry_and_output_fused(z, c_tm1)

        h = o * self.sub_activation(c)
        return h, [h, c]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))

    def get_config(self):
        config = {
            'sub_units':
            self.sub_units,
            'sub_lsmts':
            self.sub_lstms,
            'cake_activation':
            activations.serialize(self.cake_activation),
            'cake_recurrent_activation':
            activations.serialize(self.cake_recurrent_activation),
            'sub_activation':
            activations.serialize(self.sub_activation),
            'sub_recurrent_activation':
            activations.serialize(self.sub_recurrent_constraint),
            'cake_use_bias':
            self.cake_use_bias,
            'sub_use_bias':
            self.sub_use_bias,
            'cake_kernel_initializer':
            initializers.serialize(self.cake_kernel_initializer),
            'cake_recurrent_initializer':
            initializers.serialize(self.cake_recurrent_initializer),
            'cake_bias_initializer':
            initializers.serialize(self.cake_bias_initializer),
            'sub_kernel_initializer':
            initializers.serialize(self.sub_kernel_initializer),
            'sub_recurrent_initializer':
            initializers.serialize(self.sub_recurrent_initializer),
            'sub_bias_initializer':
            initializers.serialize(self.sub_bias_initializer),
            'cake_unit_forget_bias':
            self.cake_unit_forget_bias,
            'sub_unit_forget_bias':
            self.sub_unit_forget_bias,
            'cake_kernel_regularizer':
            regularizers.serialize(self.cake_kernel_regularizer),
            'cake_recurrent_regularizer':
            regularizers.serialize(self.cake_recurrent_regularizer),
            'cake_bias_regularizer':
            regularizers.serialize(self.cake_bias_regularizer),
            'sub_kernel_regularizer':
            regularizers.serialize(self.sub_kernel_regularizer),
            'sub_recurrent_regularizer':
            regularizers.serialize(self.sub_recurrent_regularizer),
            'sub_bias_regularizer':
            regularizers.serialize(self.sub_bias_regularizer),
            'cake_kernel_constraint':
            constraints.serialize(self.cake_kernel_constraint),
            'cake_recurrent_constraint':
            constraints.serialize(self.cake_recurrent_constraint),
            'cake_bias_constraint':
            constraints.serialize(self.cake_bias_constraint),
            'sub_kernel_constraint':
            constraints.serialize(self.sub_kernel_constraint),
            'sub_recurrent_constraint':
            constraints.serialize(self.sub_recurrent_constraint),
            'sub_bias_constraint':
            constraints.serialize(self.sub_bias_constraint),
            'cake_dropout':
            self.cake_dropout,
            'cake_recurrent_dropout':
            self.cake_recurrent_dropout,
            'sub_dropout':
            self.sub_dropout,
            'sub_recurrent_dropout':
            self.sub_recurrent_dropout,
            'implementation':
            self.implementation
        }
        base_config = super(JujubeCakeCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class JujubeCake(RNN):
    def __init__(self,
                 sub_units,
                 sub_lstms,

                 cake_activation='tanh',
                 cake_recurrent_activation='hard_sigmoid',
                 sub_activation='tanh',
                 sub_recurrent_activation='hard_sigmoid',

                 cake_use_bias=True,
                 sub_use_bias=True,

                 cake_kernel_initializer='glorot_uniform',
                 cake_recurrent_initializer='orthogonal',
                 cake_bias_initializer='zeros',
                 sub_kernel_initializer='glorot_uniform',
                 sub_recurrent_initializer='orthogonal',
                 sub_bias_initializer='zeros',

                 cake_unit_forget_bias=True,
                 sub_unit_forget_bias=True,

                 cake_kernel_regularizer=None,
                 cake_recurrent_regularizer=None,
                 cake_bias_regularizer=None,
                 sub_kernel_regularizer=None,
                 sub_recurrent_regularizer=None,
                 sub_bias_regularizer=None,

                 activity_regularizer=None,

                 cake_kernel_constraint=None,
                 cake_recurrent_constraint=None,
                 cake_bias_constraint=None,
                 sub_kernel_constraint=None,
                 sub_recurrent_constraint=None,
                 sub_bias_constraint=None,

                 cake_dropout=0.,
                 cake_recurrent_dropout=0.,
                 sub_dropout=0.,
                 sub_recurrent_dropout=0.,

                 implementation=1,

                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        cell = JujubeCakeCell(sub_units,
                              sub_lstms,

                              cake_activation=cake_activation,
                              cake_recurrent_activation=cake_recurrent_activation,
                              sub_activation=sub_activation,
                              sub_recurrent_activation=sub_recurrent_activation,

                              cake_use_bias=cake_use_bias,
                              sub_use_bias=sub_use_bias,

                              cake_kernel_initializer=cake_kernel_initializer,
                              cake_recurrent_initializer=cake_recurrent_initializer,
                              cake_bias_initializer=cake_bias_initializer,
                              sub_kernel_initializer=sub_kernel_initializer,
                              sub_recurrent_initializer=sub_recurrent_initializer,
                              sub_bias_initializer=sub_bias_initializer,

                              cake_unit_forget_bias=cake_unit_forget_bias,
                              sub_unit_forget_bias=sub_unit_forget_bias,

                              cake_kernel_regularizer=cake_kernel_regularizer,
                              cake_recurrent_regularizer=cake_recurrent_regularizer,
                              cake_bias_regularizer=cake_bias_regularizer,
                              sub_kernel_regularizer=sub_kernel_regularizer,
                              sub_recurrent_regularizer=sub_recurrent_regularizer,
                              sub_bias_regularizer=sub_bias_regularizer,

                              cake_kernel_constraint=cake_kernel_constraint,
                              cake_recurrent_constraint=cake_recurrent_constraint,
                              cake_bias_constraint=cake_bias_constraint,
                              sub_kernel_constraint=sub_kernel_constraint,
                              sub_recurrent_constraint=sub_recurrent_constraint,
                              sub_bias_constraint=sub_bias_constraint,

                              cake_dropout=cake_dropout,
                              cake_recurrent_dropout=cake_recurrent_dropout,
                              sub_dropout=sub_dropout,
                              sub_recurrent_dropout=sub_recurrent_dropout,

                              implementation=implementation,

                              **kwargs)
        super(JujubeCake, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        self.cell._sub_dropout_mask = None
        self.cell._sub_recurrent_dropout_mask = None
        return super(JujubeCake, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def sub_units(self):
        return self.cell.sub_units

    @property
    def sub_lstms(self):
        return self.cell.sub_lstms

    @property
    def sub_activation(self):
        return self.cell.sub_acitvation

    @property
    def cake_activation(self):
        return self.cell.cake_activation

    @property
    def sub_recurrent_activation(self):
        return self.cell.sub_recurrent_activation

    @property
    def cake_recurrent_activation(self):
        return self.cell.cake_recurrent_activation(self)

    @property
    def sub_use_bias(self):
        return self.cell.sub_use_bias

    @property
    def cake_use_bias(self):
        return self.cell.cake_use_bias

    @property
    def sub_kernel_initializer(self):
        return self.cell.sub_kernel_initializer

    @property
    def cake_kernel_initializer(self):
        return self.cell.cake_kernel_initializer

    @property
    def sub_recurrent_initializer(self):
        return self.cell.sub_recurrent_initializer

    @property
    def cake_recurrent_initializer(self):
        return self.cell.cake_recurrent_initializer

    @property
    def sub_bias_initializer(self):
        return self.cell.sub_bias_initializer

    @property
    def cake_bias_initializer(self):
        return self.cell.cake_bias_initializer

    @property
    def sub_unit_forget_bias(self):
        return self.cell.sub_unit_forget_bias

    @property
    def cake_unit_forget_bias(self):
        return self.cell.cake_unit_forget_bias

    @property
    def sub_kernel_regularizer(self):
        return self.cell.sub_kernel_regularizer

    @property
    def cake_kernel_regularizer(self):
        return self.cell.cake_kernel_regularizer

    @property
    def sub_recurrent_regularizer(self):
        return self.cell.sub_recurrent_regularizer

    @property
    def cake_recurrent_regularizer(self):
        return self.cell.cake_recurrent_regularizer

    @property
    def sub_bias_regularizer(self):
        return self.cell.sub_bias_regularizer

    @property
    def cake_bias_regularizer(self):
        return self.cell.cake_bias_regularizer

    @property
    def sub_kernel_constraint(self):
        return self.cell.sub_kernel_constraint

    @property
    def cake_kernel_constraint(self):
        return self.cell.cake_kernel_constraint

    @property
    def sub_recurrent_constraint(self):
        return self.cell.sub_recurrent_constraint

    @property
    def cake_recurrent_constraint(self):
        return self.cell.cake_recurrent_constraint

    @property
    def sub_bias_constraint(self):
        return self.cell.sub_bias_constraint

    @property
    def cake_bias_constraint(self):
        return self.cell.cake_bias_constraint

    @property
    def sub_dropout(self):
        return self.cell.sub_dropout

    @property
    def cake_dropout(self):
        return self.cell.cake_dropout

    @property
    def sub_recurrent_dropout(self):
        return self.cell.sub_recurrent_dropout

    @property
    def cake_recurrent_dropout(self):
        return self.cell.cake_recurrent_dropout

    @property
    def implementation(self):
        return self.cell.implementation

    def get_config(self):
        config = {
            'sub_units':
                self.sub_units,
                'sub_lstms':
                self.sub_lstms,
            'sub_activation':
                activations.serialize(self.sub_activation),
            'cake_activation':
                activations.serialize(self.cake_activation),
            'sub_use_bias':
                self.sub_use_bias,
            'cake_use_bias':
                self.cake_use_bias,
            'sub_kernel_initializer':
                initializers.serialize(self.sub_kernel_initializer),
            'cake_kernel_initializer':
                initializers.serialize(self.cake_kernel_initializer),
            'sub_recurrent_initializer':
                initializers.serialize(self.sub_recurrent_initializer),
            'cake_recurrent_initializer':
                initializers.serialize(self.cake_recurrent_initializer),
            'sub_bias_initializer':
                initializers.serialize(self.sub_bias_initializer),
            'cake_bias_initializer':
                initializers.serialize(self.cake_bias_initializer),
            'sub_unit_forget_bias':
                self.sub_unit_forget_bias,
            'cake_unit_forget_bias':
                self.cake_unit_forget_bias,
            'sub_kernel_regularizer':
                regularizers.serialize(self.sub_kernel_regularizer),
            'cake_kernel_regularizer':
                regularizers.serialize(self.cake_kernel_regularizer),
            'sub_recurrent_regularizer':
                regularizers.serialize(self.sub_recurrent_regularizer),
            'cake_recurrent_regularizer':
                regularizers.serialize(self.cake_recurrent_regularizer),
            'sub_bias_regularizer':
                regularizers.serialize(self.sub_bias_regularizer),
            'cake_bias_regularizer':
                regularizers.serialize(self.cake_bias_regularizer),
            'sub_activity_regularizer':
                regularizers.serialize(self.sub_activity_regularizer),
            'cake_activity_regularizer':
                regularizers.serialize(self.cake_activity_regularizer),
            'sub_kernel_constraint':
                constraints.serialize(self.sub_kernel_constraint),
            'cake_kernel_constraint':
                constraints.serialize(self.cake_kernel_constraint),
            'sub_recurrent_constraint':
                constraints.serialize(self.sub_recurrent_constraint),
            'cake_recurrent_constraint':
                constraints.serialize(self.cake_recurrent_constraint),
            'sub_bias_constraint':
                constraints.serialize(self.sub_bias_constraint),
            'cake_bias_constraint':
                constraints.serialize(self.cake_bias_constraint),
            'sub_dropout':
                self.sub_dropout,
            'cake_dropout':
                self.cake_dropout,
            'sub_recurrent_dropout':
                self.sub_recurrent_dropout,
            'cake_recurrent_dropout':
                self.cake_recurrent_dropout,
            'implementation':
                self.implementation
        }
        base_config = super(JujubeCake, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items())) + list(config.items())

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config and config['implementation'] == 0:
            config['implementation'] = 1
        return cls(**config)


class MaskReshape(Reshape):

    """This class is for reshape layer that supports masking."""

    def __init__(self, target_shape, factor, **kwargs):
        super(MaskReshape, self).__init__(target_shape, **kwargs)
        self.target_shape = tuple(target_shape)
        self.factor = factor

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            mask_shape = K.shape(mask)
            print('mask_shape', mask_shape)
            reshaped_mask = array_ops.reshape(
                mask, (K.shape(mask)[0], K.cast(K.shape(mask)[1]/self.factor, 'int32'), self.factor))
            reshaped_mask = K.cast(reshaped_mask, 'float32')
            reshaped_mean_mask = K.mean(reshaped_mask, -1, keepdims=False)
            reshaped_mean_mask = tf.math.ceil(reshaped_mean_mask)
            reshaped_mean_mask = K.cast(reshaped_mean_mask, 'bool')
            return reshaped_mean_mask
