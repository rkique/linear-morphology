import torch
import sys
sys.path.append('..')
from transformers import GPTJForCausalLM, AutoTokenizer
import lre.models as models
import lre.functional as functional
from baukit.baukit import parameter_names, get_parameter

import os
from dataclasses import dataclass, field

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

device = None
lm_params = {}

#set the device for this module.
def determine_device(mt):
    global device
    device = models.determine_device(mt)

#determine model parameters for use in LLRA below.
def determine_params(mt):
    global lm_params
    model = mt.model

    #record LayerNorms for 0-27.
    for i in range(0, 28):
        w_name = f'transformer.h.{i}.ln_1.weight'
        b_name = f'transformer.h.{i}.ln_1.bias'
        weight = get_parameter(model=model,name=w_name).data.to(device)
        bias = get_parameter(model=model,name=b_name).data.to(device)
        lm_params[w_name] = weight.to(device)
        lm_params[b_name] = bias.to(device)

    #record model LayerNorm
    ln_f_w_name = 'transformer.ln_f.weight'
    ln_f_b_name = 'transformer.ln_f.bias'
    weight = get_parameter(model=model,name=ln_f_w_name).data.to(device)
    bias = get_parameter(model=model,name=ln_f_b_name).data.to(device)
    lm_params[ln_f_w_name] = weight.to(device)
    lm_params[ln_f_b_name] = bias.to(device)

def get_position_ids(mt, prompt):
    inputs = mt.tokenizer(prompt, return_tensors="pt").to(device)
    return inputs.position_ids
    
#record a single weight and bias (for debugging purposes)
def sample_weights_biases(subject, kind, i, samples) -> dict:
    layer_dict = {"i": i}
    weights = []
    biases = []
    weight_path = f"{wdir}/{subject}/{kind}_weight_h_{i}.pt"
    bias_path = f"{wdir}/{subject}/{kind}_bias_h_{i}.pt"
    #load s_s_weight and s_s_bias
    weight = torch.load(weight_path).to(device)
    bias = torch.load(bias_path).to(device)
    layer_dict[f'{kind}_weight'] = weight
    layer_dict[f'{kind}_bias'] = bias
    return layer_dict
    
#handles either individual paths at layer i, or intervals [i,j]
def wb_paths(wdir: str, sample: str, kind: str, i: int, j: int) -> tuple:
    if j == None:
        weight_path = f"{wdir}/{sample}/{kind}_weight_{i}_{i+1}.pt"
        bias_path = f"{wdir}/{sample}/{kind}_bias_{i}_{i+1}.pt"
    else: 
        weight_path = f"{wdir}/{sample}/{kind}_weight_{i}_{j}.pt"
        bias_path = f"{wdir}/{sample}/{kind}_bias_{i}_{j}.pt"
    return (weight_path, bias_path)

#Given a working directory, 
#        the kind of derivative (S_S, S_O, or O_O),
#        samples, start
def mean_weights_biases(wdir: str, kind: str, samples: list, i, j=None) -> dict:
    layer_dict = {"i": i}
    weights = []
    biases = []
    samples = [x for x in samples if x != '.ipynb_checkpoints']
    for sample in samples:
        (weight_path, bias_path) = wb_paths(wdir, sample, kind, i, j=j)
        #load s_s_weight and s_s_bias
        weight = torch.load(weight_path).to(device)
        bias = torch.load(bias_path).to(device)
        #print(f'weight is {tp(weight)}')
        #append to lists
        weights.append(weight)
        biases.append(bias)
    mean_weight = torch.stack(weights).mean(dim=0).to(device)
    mean_bias = torch.stack(biases).mean(dim=0).to(device)
    sum_weight = torch.stack(weights).sum(dim=0).to(device)
    sum_bias = torch.stack(biases).sum(dim=0).to(device)
    
    layer_dict[f'{kind}_mean_weight'] = mean_weight
    layer_dict[f'{kind}_mean_bias'] = mean_bias
    layer_dict[f'{kind}_sum_weight'] = sum_weight
    layer_dict[f'{kind}_sum_bias'] = sum_bias
    return layer_dict
    
#for retrieving individual weights or biases (ex: building an LRE)
def mean_weight_or_bias(wdir, kind, samples):
    weights = []
    for sample in samples:
        weight_path = f"{wdir}/{sample}/{kind}.pt"
        #load s_s_weight and s_s_bias
        weight = torch.load(weight_path).to(device)
        #append to lists
        weights.append(weight)
    mean_weight = torch.stack(weights).mean(dim=0).to(device)
    return mean_weight

#Builds a layerwise LRA from saved weights.
def build_llra(wdir, samples, start, S_O_start, S_O_end, end) -> dict:
    layer_dicts = []
    ### S --> S'
    for i in range(start, S_O_start):
        layer_dict = mean_weights_biases(wdir, "s_s", samples, i)
        layer_dicts.append(layer_dict)
    
    #### S' --> O
    for i in range(S_O_start,S_O_end):
        layer_dict = mean_weights_biases(wdir, "s_o", samples, i)
        layer_dicts.append(layer_dict)
    
    ### O --> O'
    for i in range(S_O_end, end):
        layer_dict = mean_weights_biases(wdir, "o_o", samples, i)
        layer_dicts.append(layer_dict)
        
    return layer_dicts

#Build a chain (three gradient) LRA from saved weights
def build_clra(wdir, samples, S_S: tuple, S_O: tuple, O_O: tuple) -> dict:
    (S_S_start, S_S_end) = S_S
    (S_O_start, S_O_end) = S_O
    (O_O_start, O_O_end) = O_O
    layer_dicts = []
    
    ### S --> S'
    layer_dict = mean_weights_biases(wdir, "s_o", samples, S_S_start, S_S_end)
    layer_dicts.append(layer_dict)
    
    #### S' --> O
    layer_dict = mean_weights_biases(wdir, "s_o", samples, S_O_start, S_O_end)
    layer_dicts.append(layer_dict)

    ### O --> O'
    layer_dict = mean_weights_biases(wdir, "o_o", samples, O_O_start, O_O_end)
    layer_dicts.append(layer_dict)

    return layer_dicts


@dataclass(frozen=True)
class LLRA:
    """A layerwise linear relational approximator. 
    Meant to have the functions called in order, 
    # e.g. approx_s_s for layers 1-10, approx_s_o for layers 11-15, approx_o_o for layers 16-27
    """

    layer_dicts: list[dict]
    
    def get_layer_dict(self, i):
        return next((item for item in self.layer_dicts if item['i'] == i), None)
        
    def approx_s_s_layer(self, hs: torch.Tensor, i, beta=1):
        layer_dict = self.get_layer_dict(i)
        layer_weight = layer_dict['s_s_mean_weight']
        layer_bias = layer_dict['s_s_mean_bias']
        ln_weight = lm_params[f'transformer.h.{i - 1}.ln_1.weight']
        ln_bias = lm_params[f'transformer.h.{i - 1}.ln_1.bias']
        _hs = hs
    
        hs = layer_norm(hs, (1)) * ln_weight + ln_bias
        hs = beta * hs.mm(layer_weight.t()) #+ layer_bias
        #hs = hs + _hs NOTE: Residuals don't make sense with LN->full derivative
        #print(f"[{i}] S_S applied")
        return hs
        
    def approx_s_o_layer(self,hs: torch.Tensor, i, beta=1):
        layer_dict = self.get_layer_dict(i)
        layer_weight = layer_dict['s_o_mean_weight']
        layer_bias = layer_dict['s_o_mean_bias']  
        ln_weight = lm_params[f'transformer.h.{i - 1}.ln_1.weight']
        ln_bias = lm_params[f'transformer.h.{i - 1}.ln_1.bias']
        _hs = hs
        
        hs = layer_norm(hs, (1)) * ln_weight + ln_bias
        #this transformation should encompass the work of the MHSA and MLP layer.
        hs = beta * hs.mm(layer_weight.t()) + layer_bias
        #hs = hs + _hs
        #print(f"[{i}] S_O applied")
        return hs
    
    def approx_o_o_layer(self, hs: torch.Tensor, i, beta=1):
        layer_dict = self.get_layer_dict(i)
        layer_weight = layer_dict['o_o_mean_weight']
        layer_bias = layer_dict['o_o_mean_bias']  
        ln_weight = lm_params[f'transformer.h.{i - 1}.ln_1.weight']
        ln_bias = lm_params[f'transformer.h.{i - 1}.ln_1.bias']
        _hs = hs
        
        hs = layer_norm(hs, (1)) * ln_weight + ln_bias
        hs = beta * hs.mm(layer_weight.t()) #+ layer_bias
        #hs = hs + _hs
        #print(f"[{i}] O_O applied")
        return hs
        
    #Given a layerwise approximator, a hidden state, beta, and the layer to apply beta at
    #Returns the final output hidden state of the LM approximation (at the 28th layer)
    def approx_lm(self, hs: torch.Tensor, 
                  start: int, S_O_start: int, S_O_end: int, end: int,
                  beta: int = 1, beta_layer: int = None, layerwise=False):
        
        log_msg = "[approx lm] "
        if layerwise:
            for i in range(start, S_O_start):
                if i == beta_layer:
                    hs = self.approx_s_s_layer(hs, i, beta)
                else:
                    hs = self.approx_s_s_layer(hs,i)
                log_msg += f" ({i},S_S) "
            
            for i in range(S_O_start, S_O_end):
                if i == beta_layer:
                    hs = self.approx_s_o_layer(hs, i, beta)
                else:
                    hs = self.approx_s_o_layer(hs, i)
                log_msg += f" ({i},S_O) "
                
            for i in range(S_O_end, end):
                if i == beta_layer:
                    hs = self.approx_o_o_layer(hs, i, beta)
                else:
                    hs = self.approx_o_o_layer(hs, i)
                log_msg += f" ({i},O_O) "
                
        else:
            hs = self.approx_s_o_layer(hs, start, beta)
            hs = self.approx_s_o_layer(hs, S_O_start, beta)
            hs = self.approx_o_o_layer(hs, S_O_end, beta)
            log_msg += f"Applied S_S {start} S_O {S_O_start} O_O {S_O_end}"
            
        logger.info(log_msg)
        ln_weight = lm_params['transformer.ln_f.weight']
        ln_bias = lm_params['transformer.ln_f.bias']
        hs = layer_norm(hs, (1)) * ln_weight + ln_bias
        return hs

    
#Given an output hidden state
#Returns the lm's next token prediction for a (4096) embedding.
#lm_head applies LayerNorm and then a linear map to get the token-space (50400)
# (1,4096) -layernorm, linear-> (1,50400) -softmax-> (1,50400) -topk-> (1,5)
def get_object(mt, z, k=5):
    logits = mt.lm_head(z)
    dist = torch.softmax(logits.float(), dim=-1)
    topk = dist.topk(k=k, dim=-1)
    probs = topk.values.view(5).tolist()
    token_ids = topk.indices.view(5).tolist()
    words = [mt.tokenizer.decode(token_id) for token_id in token_ids]
    return (words, probs)

#returns the hidden state of subject for a prompt.
def get_hidden_state(mt, prompt, subject, h_layer):
    device = models.determine_device(mt)
    prompt = prompt.format(subject)
    h_index, inputs = functional.find_subject_token_index(
        mt = mt, prompt=prompt, subject=subject)
    #print(f'h_index is {h_index}, inputs is {inputs}')
    [[hs], _] = functional.compute_hidden_states(
        mt = mt, layers = [h_layer], inputs = inputs)
    h = hs[:, h_index]
    h = h.to(device)
    return h

#returns the hidden states for a prompt, and the subject index.
def get_hidden_states(mt, prompt, subject, h_layer):
    device = models.determine_device(mt)
    prompt = prompt.format(subject)
    h_index, inputs = functional.find_subject_token_index(
        mt = mt, prompt=prompt, subject=subject)
    token = mt.tokenizer.convert_ids_to_tokens([inputs.input_ids[:,h_index]])
    #print(f'h_index is {h_index}, token is {token}')
    [[hs], _] = functional.compute_hidden_states(
        mt = mt, layers = [h_layer], inputs = inputs)
    return hs, h_index

#This actually gets the state before the sample.object[0] token.
#This allows get_object to predict the sample.object[0] token.
def get_final_state(mt, sample, prompt):
    hs, h_index = get_hidden_states(mt, prompt, sample.object[0], 27)
    #print(f"got {sample.object[0]} from {prompt}, {h_index} out of {len(hs[0])}")
    return hs[:, h_index - 1]
    
def attn_mlp(hs, i):
    res = hs
    position_ids = torch.tensor(list(range(0, hs.shape[1]))).to(device)
    attn_outputs = mt.model.transformer.h[i].attn(hs, position_ids=position_ids)
    attn_output = attn_outputs[0]
    mlp =  mt.model.transformer.h[i].mlp(hs)
    hs = attn_output + mlp + res
    return hs

# From-scratch implementation of nn.LayerNorm
def layer_norm(x: torch.Tensor, dim, eps: float = 0.00001) -> torch.Tensor:
    mean = torch.mean(x, dim=dim, keepdim=True)
    var = torch.square(x - mean).mean(dim=dim, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)
    