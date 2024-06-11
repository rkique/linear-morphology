import sys
import json
sys.path.append('..')
from lre.data import Relation, RelationSample
from lre.operators import JacobianIclEstimator, Word2VecIclEstimator
import lre.functional as functional
import lre.models as models
import lre.metrics as metrics
import lre.logging_utils as logging_utils
from collections import defaultdict
from transformers import GPTJForCausalLM, AutoTokenizer
import torch
import random
import time
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format = logging_utils.DEFAULT_FORMAT,
    datefmt=logging_utils.DEFAULT_DATEFMT,
    stream=sys.stdout
)

counts_by_lm_correct: dict[bool, int] = defaultdict(int)

def test_operator_on_relation(operator, relation, mt, h_layer, z_layer, n_icl=8, k=5):
    logger.info(f'starting test: {operator} on {relation}')
    prompt_template = relation.prompt_templates[0]
    clozed_prompts = []
    clozed_answers = []
    #For each sample...
    for x in relation.samples:
        samples_without_x = [s for s in relation.samples if s != x]
        #assemble in-context prompt with randomly selected ICL samples
        samples_without_x = random.sample(samples_without_x, n_icl)
        #make the prompt
        cloze_template = functional.make_prompt(
            prompt_template=prompt_template,
            subject="{}",
            examples = samples_without_x
            )
        cloze_prompt = cloze_template.format(x.subject)
        clozed_prompts.append(cloze_prompt)
        clozed_answers.append(x.object)

    #should print 50 (?)
    for prompt in clozed_prompts:
        print(f'Prompt: \n{prompt}\n')

    #LM prediction
    start_time = time.time()
    logging.info(f'starting next token prediction')
    outputs_lm = functional.predict_next_token(mt=mt, prompt=clozed_prompts, k=k)
    preds_lm =  [[x.token for x in xs] for xs in outputs_lm]
    recall_lm = metrics.recall(preds_lm, clozed_answers)
    end_time = time.time()
    logging.info(f'total LM prediction time: {end_time - start_time} seconds with recall {recall_lm}')

    #operator prediction
    start_time = time.time()
    logging.info(f'starting operator prediction')
    operator = operator(mt=mt, h_layer=h_layer, z_layer=z_layer)
    operator = operator(relation)
    end_time = time.time()
    logging.info(f'total operator prediction time: {end_time - start_time} seconds')

    outputs_lre = []
    for sample in relation.samples:
        output_lre = operator(sample.subject, k=k)
        outputs_lre.append(output_lre.predictions)

    #remember that predictions is made up of (token,probs)
    preds_lre = [[x.token for x in xs] for xs in outputs_lre]
    recall_lre = metrics.recall(preds_lre, clozed_answers)

    preds_by_lm_correct = defaultdict(list)
    targets_by_lm_correct = defaultdict(list)

    #if the LM was correct, append pred_lre to preds_by_lm_correct (sth like {True: 5, False: 2})
    for pred_lm, pred_lre, target in zip(preds_lm, preds_lre, clozed_answers):
        lm_correct = metrics.any_is_nontrivial_prefix(pred_lm, target)
        preds_by_lm_correct[lm_correct].append(pred_lre)
        targets_by_lm_correct[lm_correct].append(target)
        counts_by_lm_correct[lm_correct] += 1

    print(f'For {type(operator)} on {relation.name} (out of correct, with {len(relation.samples)} total): {counts_by_lm_correct}')

json_path = "json/enckno/E01 [country - capital].json"
with open(json_path, 'r') as file:
    data = json.load(file)

relation = Relation.from_dict(data)

device = "cuda"
logging.info(f'Loading GPT-J and tokenizer')
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
logging.info('Model loaded')
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
logging.info('Tokenizer loaded')
model.to(device)
logging.info('Model put on cuda')
tokenizer.pad_token = tokenizer.eos_token
mt = models.ModelAndTokenizer(model,tokenizer)
#8 ICL examples, 50 different samples total.
test_operator_on_relation(Word2VecIclEstimator, relation, mt, 5, 27, k=5)
#test_operator_on_relation(JacobianIclEstimator, relation, mt, 5, 27, k=5)