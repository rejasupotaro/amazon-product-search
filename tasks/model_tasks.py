import onnx
import torch
from invoke import task
from transformers import AutoModel, AutoTokenizer

from amazon_product_search.constants import HF


@task
def export(c):
    bert_model_name = HF.JP_SLUKE_MEAN
    model = AutoModel.from_pretrained(bert_model_name)
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    model_name = "models/model.onnx"

    with torch.no_grad():
        sample_tokenized = tokenizer("dummy_text", return_tensors="pt")

    torch.onnx.export(
        model,
        tuple(sample_tokenized.values()),
        f=model_name,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "last_hidden_state": {0: "batch_size", 1: "sequence"},
            "pooler_output": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )

    onnx_model = onnx.load(model_name)
    print("====== Inputs ======")
    for i, input in enumerate(onnx_model.graph.input):
        print("[Input #{}]".format(i))
        print(input)

    print("====== Outputs ======")
    for i, output in enumerate(onnx_model.graph.output):
        print("[Output #{}]".format(i))
        print(output)
