import sys
import json
sys.path.append('..')
from lre.data import Relation, RelationSample, Sequence
from lre.operators import JacobianIclMeanEstimator, Word2VecIclEstimator
import lre.functional as functional
import lre.models as models
import lre.metrics as metrics
import lre.logging_utils as logging_utils
from collections import defaultdict
from transformers import GPTJForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import random
import time
import logging
import copy
import os

DEFAULT_N_ICL = 8 
device = 'cuda:1'

logger = logging.getLogger(__name__)

#(MAKE SURE TO CHANGE THIS EACH RUN)
RESULTS_FILE = 'results/Wednesday_Cuda1.txt'

logging.basicConfig(
    filename=RESULTS_FILE,
    level=logging.INFO,
    format = logging_utils.DEFAULT_FORMAT,
    datefmt=logging_utils.DEFAULT_DATEFMT,
)

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logging.info('loading model + tokenizer')
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)

model.to(device)
    
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token

mt = models.ModelAndTokenizer(model,tokenizer)

logging.info('model + tokenizer loaded')

def test_operator_on_relation(operator, relation, h_layer, z_layer):
    counts_by_lre_correct: dict[bool, int] = defaultdict(int)
    logger.info(f'starting test: {operator} on {relation.name}')
    prompt_template = relation.prompt_templates[0]
    clozed_prompts = []
    clozed_answers = []
    for x in relation.samples:
        samples = [x] + random.sample(relation.samples, DEFAULT_N_ICL - 1)
        #print(f'{samples} samples)')
        cloze_prompt = functional.make_prompt(
            template = prompt_template, 
            target = x,
            examples = samples
            )
        clozed_prompts.append(cloze_prompt)
        clozed_answers.append(x.object)

    #LM PREDICTION
    outputs_lm = functional.predict_next_token(mt=mt, prompt=clozed_prompts)
    preds_lm =  [[x.token for x in xs] for xs in outputs_lm]
    recall_lm = metrics.recall(preds_lm, clozed_answers)

    #OPERATOR PREDICTION (SAVE WEIGHTS)
    start_time = time.time()
    logging.info(f'building operator {relation.name}')
    operator = operator(mt=mt, h_layer=h_layer, z_layer=z_layer)
    operator = operator(relation)
    end_time = time.time()
    logging.info(f'total operator prediction time: {end_time - start_time} seconds')

def all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), directory)
            file_paths.append(relative_path)
    random.shuffle(file_paths)
    return file_paths
    
directory = 'json'
file_paths = all_file_paths('json')

def test_operator_on_json(operator, json_path, h_layer, z_layer):
    with open(json_path, 'r') as file:
        data = json.load(file)
        relation = Relation.from_dict(data)
        assert all(isinstance(sample, RelationSample) for sample in relation.samples)
        test_operator_on_relation(operator, relation, h_layer, z_layer)

#json_path = 'json/lexsem/L10 [antonyms - binary].json'
#json_path = 'json/lexsem/L05 [meronyms - member].json'

#test_operator_on_json(Word2VecIclEstimator, json_path, 5, 27)
#test_operator_on_json(JacobianIclMeanEstimator, json_path, 5, 27)

for json_path in file_paths:
    print(f'reading in {json_path}')
    test_operator_on_json(JacobianIclMeanEstimator, "json/"+json_path, 1, 27)
#     json_path = 'json/' + json_path
    #test_operator_on_json(Word2VecIclEstimator, json_path, 5, 27)