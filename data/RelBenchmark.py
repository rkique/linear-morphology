import sys
import json
sys.path.append('..')
from lre.data import Relation, RelationSample
from lre.operators import JacobianIclEstimator, Word2VecIclEstimator
import lre.functional as functional
import lre.models as models
import lre.metrics as metrics
from collections import defaultdict
from transformers import GPTJForCausalLM, AutoTokenizer
import torch
import random

counts_by_lm_correct: dict[bool, int] = defaultdict(int)

def test_operator_on_relation(operator, relation, mt, h_layer, z_layer, n_icl=8, k=5):
    #assemble in-context prompts with 8 ICL samples
    prompt_template = relation.prompt_templates[0]
    clozed_prompts = []
    clozed_answers = []
    for x in relation.samples:
        clozed_samples = [s for s in relation.samples if s != x]
        clozed_samples = random.sample(clozed_samples, n_icl)
        cloze_template = functional.make_prompt(
            prompt_template=prompt_template,
            subject="{}",
            examples = clozed_samples
            )
        cloze_prompt = cloze_template.format(x.subject)
        clozed_prompts.append(cloze_prompt)
        clozed_answers.append(x.object)

    for prompt in (clozed_prompts):
        print(f'Prompt: \n{prompt}\n')

    #LM prediction. max-tokens: 2048
    outputs_lm = functional.predict_next_token(mt=mt, prompt=clozed_prompts, k=k)
    preds_lm =  [[x.token for x in xs] for xs in outputs_lm]
    recall_lm = metrics.recall(preds_lm, clozed_answers)

    #operator prediction. this part is intensive.
    operator = operator(mt=mt, h_layer=h_layer, z_layer=z_layer)
    operator = operator(relation)

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
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
mt = models.ModelAndTokenizer(model,tokenizer)

#8 ICL examples, 50 different samples total.
test_operator_on_relation(Word2VecIclEstimator, relation, mt, 5, 27, k=5)
test_operator_on_relation(JacobianIclEstimator, relation, mt, 5, 27, k=5)