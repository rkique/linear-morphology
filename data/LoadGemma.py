import torch
import sys
sys.path.append('..')
from transformers import AutoTokenizer, AutoModelForCausalLM
import lre.models as models
import lre.functional as functional
import os

import torch
import json
import random
from lre.data import Relation, RelationSample, Sequence
import lre.metrics as metrics
import lre.functional as functional

device = 'cuda:0'
weights = []
biases = []
subjects = []

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

tokenizer.pad_token = tokenizer.eos_token
mt = models.ModelAndTokenizer(model,tokenizer)