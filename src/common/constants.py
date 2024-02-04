from typing import Dict, List


CANONICAL_ALPHABET = [
    'A', 'C', 'D', 'E', 'F',
    'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]
SPECIAL_SYMBOLS = ["<unk>", "<pad>", "<sos>", "<eos>"]

VOCAB = SPECIAL_SYMBOLS + CANONICAL_ALPHABET


def get_id2token() -> Dict[int, str]:
    id2token = {}
    for i, token in enumerate(VOCAB):
        id2token[i] = token
    return id2token


def get_token2id() -> Dict[str, int]:
    token2id = {}
    for i, token in enumerate(VOCAB):
        token2id[token] = i
    return token2id


def convert_seqs2ids(seqs: List[str],
                     add_sos: bool = False,
                     add_eos: bool = False,
                     max_length: int = None) -> List[List[int]]:
    token2id = get_token2id()
    indices = []
    for seq in seqs:
        ids = []
        # add <sos> id
        ids.append(token2id["<sos>"]) if add_sos else None
        for tok in seq:
            ids.append(token2id[tok])
        # add <eos> id
        ids.append(token2id["<eos>"]) if add_eos else None
        if max_length is not None:
            # add <pad> ids
            num_pad = max_length - len(ids)
            ids.extend([token2id["<pad>"] for _ in range(num_pad)])
        indices.append(ids)
    return indices


def convert_ids2seqs(ids: List[List[int]],
                     remove_special_tokens: bool = True) -> List[str]:
    id2token = get_id2token()
    seqs = []
    for indices in ids:
        seq = ''
        for i in indices:
            token = id2token[i]
            if token in SPECIAL_SYMBOLS:
                continue
            seq += token
        seqs.append(seq)
    return seqs
