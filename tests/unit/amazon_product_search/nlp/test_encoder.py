from amazon_product_search.nlp.encoder import Encoder


def test_encode():
    encoder = Encoder()
    embeddings = encoder.encode(["hello"])
    assert embeddings.shape == (1, 768)
