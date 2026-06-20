from attributions.shared.sentence import split_sentences_from_token_ids


class CharacterTokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


def test_sentence_split_preserves_token_offsets():
    text = "First step. Second step!"
    token_ids = [ord(char) for char in text]
    spans = split_sentences_from_token_ids(CharacterTokenizer(), token_ids, offset=7)

    assert [span.text for span in spans] == ["First step.", " Second step!"]
    assert spans[0].start_pos == 7
    assert spans[-1].end_pos == 7 + len(token_ids)
