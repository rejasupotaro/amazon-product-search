from amazon_product_search.encoders.sbert_encoder import SBERTEncoder


def test_encode():
    encoder = SBERTEncoder()
    embeddings = encoder.encode(["hello"])
    assert embeddings.shape == (1, 768)
