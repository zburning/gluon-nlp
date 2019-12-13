import numpy as np
import mxnet as mx
from gluonnlp.model import GELU
from mxnet.gluon import nn
import gluonnlp as nlp
import os

class BERTLayerNorm(nn.LayerNorm):
    """BERT style Layer Normalization.

    Epsilon is added inside the square root and set to 1e-12 by default.

    Inputs:
        - **data**: input tensor with arbitrary shape.
        - **out**: output tensor with the same shape as `data`.
    """

    def __init__(self, epsilon=1e-12, in_channels=0, prefix=None, params=None):
        super(BERTLayerNorm, self).__init__(epsilon=epsilon, in_channels=in_channels,
                                            prefix=prefix, params=params)

    def hybrid_forward(self, F, data, gamma, beta):
        """forward computation."""
        return F.LayerNorm(data, gamma=gamma, beta=beta, axis=self._axis, eps=self._epsilon)


class ELECTRAComponent(nlp.model.BERTModel):
    def __init__(self, encoder, vocab_size=None, token_type_vocab_size=None, units=None,
                 embed_size=None, embed_dropout=0.0, embed_initializer=None,
                 word_embed=None, token_type_embed=None, use_pooler=True, use_decoder=True,
                 use_classifier=True, use_token_type_embed=True, emb_proj=False, prefix=None, params=None):
        super(ELECTRAComponent, self).__init__(encoder, vocab_size, token_type_vocab_size, units,
                                            embed_size, embed_dropout, embed_initializer,
                                            word_embed, token_type_embed, use_pooler, use_decoder,
                                            use_classifier, use_token_type_embed, prefix, params)
        self._embed_size = embed_size
        self._emb_proj = emb_proj
        if emb_proj:
            self._emb_to_hid = nn.Dense(units, flatten=False, use_bias=False)

        if self._use_decoder:
            self.decoder = self._get_decoder_with_emb_proj(units, vocab_size, self.word_embed[0], 'decoder_')

    def _get_decoder_with_emb_proj(self, units, vocab_size, embed, prefix):
        """ Construct a decoder for the masked language model task """
        with self.name_scope():
            decoder = nn.HybridSequential(prefix=prefix)
            decoder.add(nn.Dense(units, flatten=False))
            decoder.add(GELU())
            decoder.add(BERTLayerNorm(in_channels=units))
            if self._emb_proj:
                decoder.add(nn.Dense(self._embed_size, flatten=False))
            decoder.add(nn.Dense(vocab_size, flatten=False, params=embed.collect_params()))
        assert decoder[-1].weight == list(embed.collect_params().values())[0], \
            'The weights of word embedding are not tied with those of decoder'
        return decoder

    def _encode_sequence(self, inputs, token_types, valid_length=None):
        """Generate the representation given the input sequences.

        This is used for pre-training or fine-tuning a BERT model.
        """
        # embedding
        embedding = self.word_embed(inputs)
        if self._use_token_type_embed:
            type_embedding = self.token_type_embed(token_types)
            embedding = embedding + type_embedding
        if self._emb_proj:
            embedding = self._emb_to_hid(embedding)
        # encoding
        outputs, additional_outputs = self.encoder(embedding, valid_length=valid_length)
        return outputs, additional_outputs

    def hybrid_forward(self, F, inputs, token_types, valid_length=None, masked_positions=None):
        # pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a BERT model.
        """
        outputs = []
        seq_out, attention_out = self._encode_sequence(inputs, token_types, valid_length)
        outputs.append(seq_out)

        if self.encoder._output_all_encodings:
            assert isinstance(seq_out, list)
            output = seq_out[-1]
        else:
            output = seq_out

        if attention_out:
            outputs.append(attention_out)

        if self._use_pooler:
            pooled_out = self._apply_pooling(output)
            outputs.append(pooled_out)
            if self._use_classifier:
                next_sentence_classifier_out = self.classifier(pooled_out)
                outputs.append(next_sentence_classifier_out)
        if self._use_decoder:
            decoder_out_full = self.decoder(output)
            decoder_out_masked = self._decode(F, output, masked_positions)
            outputs.append(decoder_out_full)
            outputs.append(decoder_out_masked)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]



small_ELECTRIA_generator_hparams = {
    'attention_cell': 'multi_head',
    'num_layers': 12,
    'units': 256,
    'hidden_size': 1024,
    'max_length': 512,
    'num_heads': 4,
    'scaled': True,
    'dropout': 0.1,
    'use_residual': True,
    'embed_size': 768,
    'embed_dropout': 0.1,
    'token_type_vocab_size': 2,
    'word_embed': None,
}

small_ELECTRIA_discriminator_hparams = {
    'attention_cell': 'multi_head',
    'num_layers': 12,
    'units': 768,
    'hidden_size': 3072,
    'max_length': 512,
    'num_heads': 12,
    'scaled': True,
    'dropout': 0.1,
    'use_residual': True,
    'embed_size': 768,
    'embed_dropout': 0.1,
    'token_type_vocab_size': 2,
    'word_embed': None,
}


def get_model(hparams, dataset_name=None, vocab=None,
                   use_pooler=True, use_decoder=True, use_classifier=False, output_attention=False,
                   output_all_encodings=False, use_token_type_embed=True, emb_proj=True,
                   root=os.path.join(nlp.base.get_home_dir(), 'models'), word_embed=None,
                    **kwargs):
    predefined_args = hparams
    mutable_args = ['use_residual', 'dropout', 'embed_dropout', 'word_embed']
    mutable_args = frozenset(mutable_args)
    assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
        'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    # encoder
    encoder = nlp.model.BERTEncoder(attention_cell=predefined_args['attention_cell'],
                          num_layers=predefined_args['num_layers'],
                          units=predefined_args['units'],
                          hidden_size=predefined_args['hidden_size'],
                          max_length=predefined_args['max_length'],
                          num_heads=predefined_args['num_heads'],
                          scaled=predefined_args['scaled'],
                          dropout=predefined_args['dropout'],
                          output_attention=output_attention,
                          output_all_encodings=output_all_encodings,
                          use_residual=predefined_args['use_residual'],
                          activation=predefined_args.get('activation', 'gelu'),
                          layer_norm_eps=predefined_args.get('layer_norm_eps', None))


    # bert_vocab
    from gluonnlp.vocab import BERTVocab
    bert_vocab = nlp.model.bert._load_vocab(dataset_name, vocab, root, cls=BERTVocab)
    # BERT
    net = ELECTRAComponent(encoder, len(bert_vocab),
                    token_type_vocab_size=predefined_args['token_type_vocab_size'],
                    units=predefined_args['units'],
                    embed_size=predefined_args['embed_size'],
                    embed_dropout=predefined_args['embed_dropout'],
                    use_pooler=use_pooler, use_decoder=use_decoder,
                    use_classifier=use_classifier,
                    use_token_type_embed=use_token_type_embed,
                    emb_proj=emb_proj,
                    word_embed=word_embed)

    return net, bert_vocab

class ELECTRA(mx.gluon.HybridBlock):
    def __init__(self, generator, discriminator, units=768, dropout=0, prefix=None, params=None):
        super(ELECTRA, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._G = generator
            self._D = discriminator
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=1, flatten=False))
            self.pooler = nn.Dense(units=units, flatten=False, activation='tanh', prefix=prefix)

    def __call__(self, inputs, token_types, valid_length=None, masked_positions=None):
        return super(ELECTRA, self).__call__(inputs, token_types, valid_length, masked_positions)

    def hybrid_forward(self, F, inputs, token_types, valid_length=None, masked_positions=None):
        gen_output, decoded_full, decoded_masked = self._G(inputs, token_types, valid_length, masked_positions)
        #Considering using sampling here?
        disc_input = F.argmax(decoded_full, axis=-1)
        disc_label = F.equal(disc_input.astype('int32'), inputs)
        disc_output = self._D(disc_input, token_types, valid_length)
        classified = self.classifier(self.pooler(disc_output)).squeeze(-1)
        #print(decoded_masked, classified, decoded_full, gen_output, disc_output)
        return (decoded_masked, classified, disc_label, decoded_full, gen_output, disc_output)




def get_ELECTRA_for_pretrain(ctx, dataset_name=None, prefix=None, params=None):
        generator, vocab_g = get_model(small_ELECTRIA_generator_hparams, dataset_name=dataset_name,
                                       use_classifier=False, use_pooler=False, ctx=ctx, emb_proj=True)

        discriminator, _ = get_model(small_ELECTRIA_discriminator_hparams, dataset_name=dataset_name,
                                           use_decoder=False, use_classifier=False, use_pooler=False, ctx=ctx,
                                            emb_proj=False, word_embed=generator.word_embed)
        net = ELECTRA(generator, discriminator)
        net.initialize(init=mx.init.Normal(0.02), ctx=ctx)

        #net.hybridize(static_alloc=True)

        #mlm_loss.hybridize(static_alloc=True, static_shape=True)
        #disc_loss.hybridize(static_alloc=True, static_shape=True)

        model = ELECTRIAForPretrain(net, len(vocab_g))
        return model, vocab_g


class ELECTRIAForPretrain(mx.gluon.HybridBlock):
    def __init__(self, electra, vocab_size, prefix=None, params=None):
        super(ELECTRIAForPretrain, self).__init__(prefix=prefix, params=params)
        self._electra = electra
        self._mlm_loss = mx.gluon.loss.SoftmaxCELoss()
        self._disc_loss = mx.gluon.loss.SigmoidBCELoss()
        self._vocab_size = vocab_size

    def __call__(self, input_id, masked_id, masked_position, masked_weight, segment_id=None, valid_length=None):
        return super(ELECTRIAForPretrain, self).__call__(input_id, masked_id, masked_position,
                                                  masked_weight, segment_id, valid_length)

    def _padding_mask(self, F, inputs, valid_length):
        steps = F.contrib.arange_like(inputs, axis=1)
        mask = F.broadcast_lesser(F.reshape(steps, shape=(1, -1)),
                                   F.reshape(valid_length, shape=(-1, 1)))

        return mask

    def hybrid_forward(self, F, input_id, masked_id, masked_position, masked_weight, segment_id, valid_length):
        # pylint: disable=arguments-differ
        num_masks = masked_weight.sum() + 1e-8
        valid_length = valid_length.reshape(-1)
        masked_id = masked_id.reshape(-1)
        decoded, classified, disc_label, _, _, _ = self._electra(input_id, segment_id, valid_length, masked_position)
        decoded = decoded.reshape((-1, self._vocab_size))
        ls1 = self._mlm_loss(decoded.astype('float32', copy=False),
                            masked_id, masked_weight.reshape((-1, 1)))

        loss_mask = self._padding_mask(F, classified, valid_length)
        ls2 = self._disc_loss(classified.astype('float32'), disc_label.astype('float32'), loss_mask)
        ls1 = ls1.sum() / num_masks
        ls2 = ls2.mean() / valid_length.sum()

        return decoded, classified, disc_label, ls1, ls2
