from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

SPECIAL = ["<pad>", "<bos>", "<eos>", "<sep>"]

class BPETokenizer:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")
        self.sep_id = tokenizer.token_to_id("<sep>")
        self.vocab_size = tokenizer.get_vocab_size()

    @classmethod
    def train(cls, files, vocab_size=4096, min_frequency=2):
        tok = Tokenizer(BPE(unk_token=None))
        # ByteLevel guarantees coverage of Hebrew and mixed text without UNK.
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=True,
        )
        tok.train(files=[str(x) for x in files], trainer=trainer)
        return cls(tok)

    @classmethod
    def load(cls, path):
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path):
        self.tok.save(str(path))

    def encode(self, text, bos=True, eos=True):
        ids = self.tok.encode(text).ids
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        special = {self.pad_id, self.bos_id, self.eos_id, self.sep_id}
        return self.tok.decode([i for i in ids if i not in special])
