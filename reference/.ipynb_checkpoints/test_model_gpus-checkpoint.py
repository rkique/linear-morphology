import torch
import sys
sys.path.append('..')
from transformers import GPTJForCausalLM, AutoTokenizer
import lre.models as models
import lre.functional as functional
import os

import json
import random
from lre.data import Relation, RelationSample, Sequence
import lre.metrics as metrics
import lre.functional as functional

device = "cuda:1"
weights = []
biases = []
subjects = []
print("loading model")
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
print("model loaded")
model.to('cuda:1')
print("model moved to cuda")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token
mt = models.ModelAndTokenizer(model,tokenizer)
print("model+tokenizer loaded")