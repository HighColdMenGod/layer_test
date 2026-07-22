import pytest

from insufficient_shift.scoring import _single_token_id


class FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text, add_special_tokens=False):
        return self.mapping[text]


def test_accepts_exactly_one_token():
    tokenizer = FakeTokenizer({" A": [10]})
    assert _single_token_id(tokenizer, " A", role="Sufficient") == 10


def test_rejects_multi_token_verbalizer():
    tokenizer = FakeTokenizer({" A": [10, 11]})
    with pytest.raises(ValueError, match="not one"):
        _single_token_id(tokenizer, " A", role="Sufficient")
