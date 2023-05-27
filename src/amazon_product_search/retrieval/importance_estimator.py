import numpy as np
import torch

from amazon_product_search.modules.colbert import ColBERTWrapper


class ColBERTTermImportanceEstimator(ColBERTWrapper):
    def estimate(self, text: str) -> list[tuple[str, float]]:
        with torch.no_grad():
            tokenized_text = self.tokenize([text])
            _, _, _, stopword_importance = self.colberter.encode_doc(tokenized_text)

            results = list(
                zip(self.tokenizer.tokenize(text), stopword_importance[0], strict=True)
            )

        outputs = []
        i = 0
        while i < len(results):
            tokens, scores = [results[i][0]], [results[i][1].numpy()]
            j = i + 1
            while j < len(results):
                another_token, another_score = results[j][0], results[j][1]
                if not another_token.startswith("##"):
                    break
                another_token = another_token[2:]
                tokens.append(another_token)
                scores.append(another_score.numpy())
                j += 1
            outputs.append(("".join(tokens), np.mean(scores)))
            i = j
        return outputs
