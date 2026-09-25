from architecture import Qwen3_1_7B, make_qwen_1_7
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_weights import load_weights
from run_tokenization import encode, decode

def greedy_batch_decode(model, inputs:list[str])->list[str]: 
    msgs = []
    for input in inputs:
        msgs.append([{"role": "user", "content": input}])
    token_ids, _ = encode(msgs)
    raw = model.forward(token_ids.to("mps"))
    outputs = []
    for dist in raw:
        output_token = torch.argmax(dist)
        outputs.append(decode(output_token))
    return outputs

if __name__=="__main__":
    model = make_qwen_1_7()
    model.load_state_dict(load_weights(device="cpu"))
    print(greedy_batch_decode(model, ["my name is: "]))

