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

logger = logging.getLogger(__name__)

RESULTS_FILE = "RelBenchmark_log.txt"
LOGGING_FILE = "RelBenchmark_log_extended.txt"

logging.basicConfig(
    filename=LOGGING_FILE,
    level=logging.INFO,
    format = logging_utils.DEFAULT_FORMAT,
    datefmt=logging_utils.DEFAULT_DATEFMT,
)

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

counts_by_lre_correct: dict[bool, int] = defaultdict(int)

logging.info('loading model + tokenizer')
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)

model.to('cuda:1')
    
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token

mt = models.ModelAndTokenizer(model,tokenizer)

logging.info('model + tokenizer loaded')

def test_operator_on_relation(operator, relation, h_layer, z_layer):
    logger.info(f'starting test: {operator} on {relation}')
    prompt_template = relation.prompt_templates[0]
    clozed_prompts = []
    clozed_answers = []
    for x in relation.samples:
        samples = [x] + random.sample(relation.samples, DEFAULT_N_ICL - 1)
        print(f'{samples} samples)')
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

    #OPERATOR PREDICTION
    start_time = time.time()
    logging.info(f'starting operator prediction')
    operator = operator(mt=mt, h_layer=h_layer, z_layer=z_layer)
    operator = operator(relation)
    end_time = time.time()
    logging.info(f'total operator prediction time: {end_time - start_time} seconds')

    outputs_lre = []
    for sample in relation.samples:
        output_lre = operator(sample)
        outputs_lre.append(output_lre.predictions)

    #remember that predictions is made up of (token,probs)
    preds_lre = [[x.token for x in xs] for xs in outputs_lre]
    recall_lre = metrics.recall(preds_lre, clozed_answers)

    preds_by_lre_correct = defaultdict(list)
    targets_by_lre_correct = defaultdict(list)

    log_msg = ""
    
    for pred_lm, pred_lre, target in zip(preds_lm, preds_lre, clozed_answers):
        lm_correct = metrics.any_is_nontrivial_prefix(pred_lm, target)
        if lm_correct:
          lre_correct = metrics.any_is_nontrivial_prefix(pred_lre, target)
          log_target = f'{pred_lre} matches {target} is {lre_correct}'
          logging.info(log_target)
          log_msg += (log_target + "\n")
          preds_by_lre_correct[lre_correct].append(pred_lre)
          targets_by_lre_correct[lre_correct].append(target)
          counts_by_lre_correct[lre_correct] += 1
            
    correct_lre_ct = counts_by_lre_correct.get(True, 0)
    incorrect_lre_ct = counts_by_lre_correct.get(False, 0)
    log_overall = f'{relation.name},{len(relation.samples)},{correct_lre_ct},{incorrect_lre_ct}\n'
    
    logging.info(log_overall)
    
    with open(RESULTS_FILE, "a+") as file:
        file.write(log_msg)
        file.write(log_overall)

def all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), directory)
            file_paths.append(relative_path)
    return file_paths
    
directory = 'json'
file_paths = all_file_paths('json')
def test_operator_on_json(operator, json_path, h_layer, z_layer):
    with open(json_path, 'r') as file:
        data = json.load(file)
        relation = Relation.from_dict(data)
        assert all(isinstance(sample, RelationSample) for sample in relation.samples)
        test_operator_on_relation(operator, relation, 5, 27)

json_path = 'json/enckno/E06 [animal - youth].json'
#test_operator_on_json(Word2VecIclEstimator, json_path, 5, 27)
test_operator_on_json(JacobianIclMeanEstimator, json_path, 5, 27)

# for json_path in file_paths:
#     json_path = 'json/' + json_path
#     #test_operator_on_json(Word2VecIclEstimator, json_path, 5, 27)
#     test_operator_on_json(JacobianIclMeanEstimator, json_path, 5, 27)
