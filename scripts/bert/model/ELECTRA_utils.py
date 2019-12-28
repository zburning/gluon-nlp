import mxnet as mx
from gluonnlp.model import GELU
from mxnet.gluon import nn
import gluonnlp as nlp
import os


class ELECTRAComponent(nlp.model.BERTModel):
    def __init__(self, encoder, vocab_size=None, token_type_vocab_size=None, units=None,
                 embed_size=None, embed_initializer=None,
                 word_embed=None, token_type_embed=None, use_pooler=True, use_decoder=True,
                 use_classifier=True, use_token_type_embed=True, emb_proj=False, prefix=None, params=None):
        super(ELECTRAComponent, self).__init__(encoder, vocab_size, token_type_vocab_size, units,
                                            embed_size, embed_initializer,
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
            decoder.add(nn.LayerNorm(in_channels=units, epsilon=1e-12))
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
    'token_type_vocab_size': 2,
    'word_embed': None,
     'layer_norm_eps': 1e-12
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
    'token_type_vocab_size': 2,
    'word_embed': None,
    'layer_norm_eps': 1e-12
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
                    use_pooler=use_pooler, use_decoder=use_decoder,
                    use_classifier=use_classifier,
                    use_token_type_embed=use_token_type_embed,
                    emb_proj=emb_proj,
                    word_embed=word_embed)

    return net, bert_vocab

class ELECTRA(mx.gluon.Block):
    def __init__(self, generator, discriminator, sampling=True, units=768, dropout=0, prefix=None, params=None):
        super(ELECTRA, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._G = generator
            self._D = discriminator
            self._sampling = sampling
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=1, flatten=False))
            self.pooler = nn.Dense(units=units, flatten=False, activation='tanh', prefix=prefix)

    def __call__(self, input_orig, inputs, token_types, valid_length=None, masked_positions=None, mp_mask=None):
        return super(ELECTRA, self).__call__(input_orig, inputs, token_types, valid_length, masked_positions, mp_mask)

    def forward(self, inputs_orig, inputs, token_types, valid_length=None, masked_positions=None, mp_mask=None):
        gen_output, decoded_full, decoded_masked = self._G(inputs, token_types, valid_length, masked_positions)
        #Considering using sampling here?
        F = mx.ndarray
        disc_sampled = F.argmax(decoded_full, axis=-1) if not self._sampling \
            else F.random.multinomial(F.softmax(decoded_full), get_prob=False)

        disc_sampled = disc_sampled.as_in_context(decoded_full.context)
        disc_sampled = disc_sampled.detach()
        disc_input = inputs_orig * mp_mask + disc_sampled * (1 - mp_mask)
        disc_label = disc_input.astype('int32').__eq__(inputs_orig)
        disc_output = self._D(disc_input, token_types, valid_length)
        classified = self.classifier(self.pooler(disc_output)).squeeze(-1)
        return (decoded_masked, classified, disc_label, decoded_full, gen_output, disc_output)




def get_ELECTRA_for_pretrain(ctx, dataset_name=None, sampling=True, prefix=None, params=None):
        generator, vocab_g = get_model(small_ELECTRIA_generator_hparams, dataset_name=dataset_name,
                                       use_classifier=False, use_pooler=False, ctx=ctx, emb_proj=True)

        discriminator, _ = get_model(small_ELECTRIA_discriminator_hparams, dataset_name=dataset_name,
                                           use_decoder=False, use_classifier=False, use_pooler=False, ctx=ctx,
                                            emb_proj=False, word_embed=generator.word_embed)
        net = ELECTRA(generator, discriminator)
        net.initialize(init=mx.init.Normal(0.02), ctx=ctx)

        model = ELECTRIAForPretrain(net, len(vocab_g))
        model.hybridize(static_alloc=True)
        return model, vocab_g


class ELECTRIAForPretrain(mx.gluon.Block):
    def __init__(self, electra, vocab_size, prefix=None, params=None):
        super(ELECTRIAForPretrain, self).__init__(prefix=prefix, params=params)
        self._electra = electra
        self._mlm_loss = mx.gluon.loss.SoftmaxCELoss()
        self._disc_loss = mx.gluon.loss.SigmoidBCELoss()
        self._vocab_size = vocab_size

    def __call__(self, input_id_orig, input_id, masked_id, masked_position, mp_mask, masked_weight, segment_id=None,
                 sp_tokens_mask=None, valid_length=None):
        return super(ELECTRIAForPretrain, self).__call__(input_id_orig, input_id, masked_id, masked_position, mp_mask,
                                                  masked_weight, segment_id, sp_tokens_mask, valid_length)

    def forward(self, input_id_orig, input_id, masked_id, masked_position, mp_mask,
                       masked_weight, segment_id, sp_mask, valid_length):
        # pylint: disable=arguments-differ
        num_masks = masked_weight.sum() + 1e-8
        valid_tokens = sp_mask.sum() + 1e-8
        valid_length = valid_length.reshape(-1)
        masked_id = masked_id.reshape(-1)
        decoded, classified, disc_label, _, _, _ = self._electra(input_id_orig, input_id, segment_id, valid_length,
                                                                 masked_position, mp_mask)
        decoded = decoded.reshape((-1, self._vocab_size))
        ls1 = self._mlm_loss(decoded.astype('float32'),
                            masked_id, masked_weight.reshape((-1, 1)))

        ls2 = self._disc_loss(classified.astype('float32'), disc_label.astype('float32'), sp_mask.astype('float32'))
        ls1 = ls1.sum() / num_masks
        ls2 = ls2.sum() / valid_tokens.astype('float32')
        return decoded, classified, disc_label, ls1, ls2
