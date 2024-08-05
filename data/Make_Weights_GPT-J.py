#This script was used to generate weight and bias Jacobians for GPT-J.
#Given: folder paths and device number
#Utilizes functions in lre/operators.py and lre/functional.py to write Jacobians to file.
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
DEVICE_NUM = 1
device = f'cuda:{DEVICE_NUM}'

logger = logging.getLogger(__name__)

#(MAKE SURE TO CHANGE THIS EACH RUN)
RESULTS_FILE = f'results/Thursday_{DEVICE_NUM}_MakeWeights_W2V_6_27.txt'

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

def train_operator_on_relation(operator, relation, h_layer, z_layer):
    logger.info(f'storing weights: {operator} on {relation.name}')
    prompt_template = relation.prompt_templates[0]

    #SAVE OPERATOR WEIGHTS
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
# file_paths = all_file_paths('json')

def train_operator_on_json(operator, json_path, h_layer, z_layer):
    with open(json_path, 'r') as file:
        data = json.load(file)
        relation = Relation.from_dict(data)
        assert all(isinstance(sample, RelationSample) for sample in relation.samples)
        train_operator_on_relation(operator, relation, h_layer, z_layer)

#json_path = 'json/lexsem/L10 [antonyms - binary].json'
#json_path = 'json/lexsem/L05 [meronyms - member].json'

#test_operator_on_json(Word2VecIclEstimator, json_path, 5, 27)
#test_operator_on_json(JacobianIclMeanEstimator, json_path, 5, 27)

file_paths = [
  'json/enckno/E03 [UK_city - county].json',
 'json/enckno/E06 [animal - youth].json',
 'json/enckno/E07 [animal - sound].json',
 'json/enckno/E09 [things - color].json',
 'json/enckno/E01 [country - capital].json',
 'json/enckno/E05 [name - occupation].json',
 'json/enckno/E10 [male - female].json',
 'json/enckno/E08 [animal - shelter].json',
 'json/enckno/E02 [country - language].json',
 'json/enckno/E04 [name - nationality].json',
 'json/dermor/D10 [verb+ment_irreg].json',
 'json/dermor/D01 [noun+less_reg].json',
 'json/dermor/D05 [adj+ness_reg].json',
 'json/dermor/D06 [re+verb_reg].json',
 'json/dermor/D02 [un+adj_reg].json',
 'json/dermor/D07 [verb+able_reg].json',
 'json/dermor/D09 [verb+tion_irreg].json',
 'json/dermor/D03 [adj+ly_reg].json',
 'json/dermor/D04 [over+adj_reg].json',
 'json/dermor/D08 [verb+er_irreg].json',
 'json/infmor/I04 [adj - superlative].json',
 'json/infmor/I10 [verb_3pSg - Ved].json',
 'json/infmor/I01 [noun - plural_reg].json',
 'json/infmor/I08 [verb_Ving - 3pSg].json',
 'json/infmor/I05 [verb_inf - 3pSg].json',
 'json/infmor/I07 [verb_inf - Ved].json',
 'json/infmor/I09 [verb_Ving - Ved].json',
 'json/infmor/I06r [Ving - verb_inf].json',
 'json/infmor/I02 [noun - plural_irreg].json',
 'json/lexsem/L05 [meronyms - member].json',
 'json/lexsem/L10 [antonyms - binary].json',
 'json/lexsem/L03 [hyponyms - misc].json',
 'json/lexsem/L01 [hypernyms - animals].json',
 'json/lexsem/L07 [synonyms - intensity].json',
 'json/lexsem/L04 [meronyms - substance].json',
 'json/lexsem/L02 [hypernyms - misc].json',
 'json/lexsem/L08 [synonyms - exact].json',
 'json/lexsem/L06 [meronyms - part].json'
]

#The number parameters here are obsolete, functional.py specifies the weights made.
for json_path in file_paths:
    print(f'reading in {json_path}')
    #train_operator_on_json(Word2VecIclEstimator, json_path, 6,27)
    train_operator_on_json(JacobianIclMeanEstimator, json_path, 1, 27)