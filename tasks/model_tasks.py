import onnx
import torch
from invoke import task
from onnx import ModelProto
from onnxruntime import InferenceSession
from onnxruntime.quantization import quantize_dynamic
from torch import Tensor
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel
from transformers.models.bert.configuration_bert import BertConfig

from amazon_product_search.constants import HF, MODELS_DIR


class MeanPoolingEncoderONNX(BertPreTrainedModel):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor | None = None) -> Tensor:
        last_hidden_state = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        attention_mask = attention_mask.unsqueeze(dim=-1)
        last_hidden_state = last_hidden_state * attention_mask
        mean_vector = last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1)
        return mean_vector


def print_input_and_output_names(onnx_model: ModelProto) -> None:
    for i, input in enumerate(onnx_model.graph.input):
        print("[Input #{}]".format(i))
        print(input)

    for i, output in enumerate(onnx_model.graph.output):
        print("[Output #{}]".format(i))
        print(output)


@task
def export(c, hf_model_name=HF.EN_ALL_MINILM, output_model_name="model"):
    torch_model = MeanPoolingEncoderONNX.from_pretrained(hf_model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    onnx_model_filepath = f"{MODELS_DIR}/{output_model_name}.onnx"
    quantized_onnx_model_filepath = f"{MODELS_DIR}/{output_model_name}_quantized.onnx"

    with torch.no_grad():
        encoded_text = tokenizer("dummy_text", return_tensors="pt")

    torch.onnx.export(
        torch_model,
        args=tuple(encoded_text.values()),
        f=onnx_model_filepath,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "vector": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )

    quantize_dynamic(
        model_input=onnx_model_filepath,
        model_output=quantized_onnx_model_filepath,
    )

    print(f"====== {onnx_model_filepath} ======")
    onnx_model = onnx.load(onnx_model_filepath)
    print_input_and_output_names(onnx_model)

    print(f"====== {quantized_onnx_model_filepath} ======")
    quantized_onnx_model = onnx.load(quantized_onnx_model_filepath)
    print_input_and_output_names(quantized_onnx_model)

    session = InferenceSession(quantized_onnx_model_filepath)
    encoded_text = tokenizer(
        "dummy_text",
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    mean_vector = session.run(
        output_names=["output"],
        input_feed={
            "input_ids": encoded_text["input_ids"],
            "attention_mask": encoded_text["attention_mask"],
        },
    )
    print(mean_vector)
