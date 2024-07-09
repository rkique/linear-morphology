import torch
import llra.build as build
import sys
sys.path.append('..')
import lre.models as models
from pathlib import Path

import time
import logging

DEFAULT_FORMAT = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format = DEFAULT_FORMAT,
    datefmt=DEFAULT_DATEFMT,
)

logger = logging.getLogger(__name__)
#logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

#Given modeltokenizer, prompt, subject, and layer
#Returns a dictionary with attn_mlp weights and biases, and saves them to file
def get_jacobians(mt, 
                  relation_name: str,
                  prompt: str, 
                  subject: str, 
                  i: int):
    
    device = models.determine_device(mt)
    layer_dict = {}
    
    #(1,10,4096)
    h_index, hs = build.get_hidden_states(mt, prompt, subject, i)

    #the next hidden state is used only to estimate the biases.
    _, hs1 = build.get_hidden_states(mt, prompt, subject, i+1)
    
    o_hs = hs[:, -1, :]
    s_hs = hs[:, h_index, :]
    o_hs1 = hs1[:, -1, :]
    s_hs1 = hs1[:, h_index, :]
    
    z_index = -1
    
    hs = build.layer_norm(hs, (1))
    logging.info(f'{hs.shape=}')
    
    #this could be useful.
    #computes attention and mlp on the hidden state
    def attn_mlp(hs):
        res = hs
        position_ids = torch.tensor(list(range(0, hs.shape[1]))).to(device)
        attn = mt.model.transformer.h[i].attn.forward(hs, position_ids=position_ids)[0]
        mlp =  mt.model.transformer.h[i].mlp.forward(hs)
        hs = attn + mlp + res
        return hs
        
    start_time = time.time()
    logging.info(f"computing jacobian for layer {i} for {relation_name}.{subject}")
    j = torch.autograd.functional.jacobian(attn_mlp, hs).half().to(device)
    logging.info(f"finished layer {i} for {relation_name}.{subject}")
    s_o_weight = j[:, z_index, :, :, h_index, :].squeeze().to(device)
    s_s_weight = j[:, h_index, :, :, h_index, :].squeeze().to(device)
    o_o_weight = j[:, z_index, :, :, z_index, :].squeeze().to(device)
    end_time = time.time()
    logging.info(f'total layer calculation time: {end_time - start_time} seconds')

    s_o_bias = o_hs1 - s_hs.mm(s_o_weight.t())
    s_s_bias = s_hs1 - s_hs.mm(s_s_weight.t())
    o_o_bias = o_hs1 - o_hs.mm(o_o_weight.t())
    
    logging.info(f"""[get_jacobians] weight calculation finished \n
                s_o: {s_o_weight} \n
                s_s: {s_s_weight} \n
                o_o: {o_o_weight} \n
            """)

    directory = Path(f'vapprox/{relation_name}/{subject}')
    
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
                    
    torch.save(s_o_weight, f'vapprox/{relation_name}/{subject}/s_o_weight_{i}_{i+1}.pt')
    torch.save(s_s_weight, f'vapprox/{relation_name}/{subject}/s_s_weight_{i}_{i+1}.pt')
    torch.save(o_o_weight, f'vapprox/{relation_name}/{subject}/o_o_weight_{i}_{i+1}.pt')

    torch.save(s_o_bias, f'vapprox/{relation_name}/{subject}/s_o_bias_{i}_{i+1}.pt')
    torch.save(s_s_bias, f'vapprox/{relation_name}/{subject}/s_s_bias_{i}_{i+1}.pt')
    torch.save(o_o_bias, f'vapprox/{relation_name}/{subject}/o_o_bias_{i}_{i+1}.pt')
    logging.info("saved to file")