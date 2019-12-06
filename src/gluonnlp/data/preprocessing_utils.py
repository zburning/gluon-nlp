"""Utility classes and functions for data processing"""

__all__ = [
    'truncate_seqs_equal', 'truncate_equal_by_len', 'ConcatSeqTransform', 'TokenizeAndPositionAlign',
    'get_doc_spans', 'align_position2doc_spans', 'improve_answer_span', 'check_is_max_context'
]

import collections
import itertools
import numpy.ma as ma


def truncate_equal_by_len(lens, max_len):
    if sum(lens) <= max_len:
        return lens

    lens = ma.masked_array(lens, mask=[0] * len(lens))
    while True:
        argmin = lens.argmin()
        minval = lens[argmin]
        quotient, remainder = divmod(max_len, len(lens) - sum(lens.mask))
        if minval <= quotient:  # Ignore values that don't need truncation
            lens.mask[argmin] = 1
            max_len -= minval
        else:  # Truncate all
            lens.data[~lens.mask] = [
                quotient + 1 if i < remainder else quotient
                for i in range(lens.count())
            ]
            break

    return lens.data.tolist()


def truncate_seqs_equal(seqs, max_len):
    """
    truncate a list of seqs so that the total length equals max length.
    Trying to truncate the seqs to equal length.

    Returns
    -------
    list : list of truncated sequence keeping the origin order
    """
    assert isinstance(seqs, list)
    lens = list(map(len, seqs))
    seqs = [seq[:length] for (seq, length) in zip(seqs, truncate_equal_by_len(lens, max_len))]
    return seqs


def ConcatSeqTransform(seqs, separators, separator_mask=1):
    """
    Insert special tokens for sequence list or a single sequence.
    For sequence pairs, the input is a list of 2 strings:
    text_a, text_b.
    Inputs:
       text_a: 'is this jacksonville ?'
       text_b: 'no it is not'
       separator: [[SEP], [SEP]]

    Processed:
       tokens:     'is this jacksonville ? [SEP] no it is not . [SEP]'
       segment_ids: 0  0    0            0  0    1  1  1  1   1 1
       p_mask:      0  0    0            0  1    0  0  0  0   0 1
       valid_length: 11

    Parameters
    ----------
    separator : list
        The special tokens to be appended to each sequence. For example:
        Given:
            seqs: [[1, 2], [3, 4], [5, 6]]
            separator: [[], 7]
        it will be:
            [1, 2, 3, 4, 7, 5, 6]

    seqs : list of sequences or a single sequence

    Returns
    -------
    np.array: input token ids in 'int32', shape (batch_size, seq_length)
    np.array: segment ids in 'int32', shape (batch_size, seq_length)
    np.array: mask for special tokens
    """
    assert isinstance(seqs, collections.abc.Iterable) and len(seqs) > 0
    concat = sum((seq + sep for sep, seq in
                  itertools.zip_longest(separators, seqs, fillvalue=[])), [])
    segment_ids = sum(([i] * (len(seq) + len(sep)) for i, (sep, seq) in
                       enumerate(itertools.zip_longest(separators, seqs, fillvalue=[]))), [])
    p_mask = sum(([0] * len(seq) + [separator_mask] * len(sep) for sep, seq in
                  itertools.zip_longest(separators, seqs, fillvalue=[])), [])
    return concat, segment_ids, p_mask


def TokenizeAndPositionAlign(origin_text, positions, tokenizer):
    """Tokenize the text and align the origin positions to the corresponding position"""
    if not isinstance(positions, list):
        positions = [positions]
    orig_to_tok_index = []
    tokenized_text = []
    for (i, token) in enumerate(origin_text):
        orig_to_tok_index.append(len(tokenized_text))
        sub_tokens = tokenizer(token)
        tokenized_text += sub_tokens
    new_positions = [orig_to_tok_index[p] for p in positions]
    return new_positions, tokenized_text


def get_doc_spans(full_doc, max_length, doc_stride):
    """A simple function that applying a sliding window on the doc and get doc spans

     Parameters
    ----------
    full_doc: list
        The origin doc text
    max_length: max_length
        Maximum size of a doc span
    doc_stride: int
        Step of sliding window

    Returns
    -------
    list: a list of processed doc spans
    list: a list of start/end index of each doc span
    """
    doc_spans = []
    start_offset = 0
    while start_offset < len(full_doc):
        length = min(max_length, len(full_doc) - start_offset)
        end_offset = start_offset + length
        doc_spans.append((full_doc[start_offset: end_offset], (start_offset, end_offset)))
        start_offset += min(length, doc_stride)
    return list(zip(*doc_spans))


def align_position2doc_spans(positions, doc_spans_indices,
                             offset=0, default_value=-1, all_in_span=True):
    """Align the origin positions to the corresponding position in doc spans"""
    if not isinstance(positions, list):
        positions = [positions]
    doc_start, doc_end = doc_spans_indices
    if all_in_span and not all([p in range(doc_start, doc_end + 1) for p in positions]):
        return [default_value] * len(positions)
    new_positions = [p - doc_start + offset if p in range(doc_start, doc_end + 1)
                     else default_value for p in positions]
    return new_positions


def improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                        orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = ' '.join(tokenizer(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = ' '.join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
