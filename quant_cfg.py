from hqq.core.quantize import BaseQuantizeConfig

# TODO: Make your own quant config for DeiT-S
def get_quant_config_deit(model):
    quant_config = {}
    
    n_blocks = len(model.blocks)
    q6_config = BaseQuantizeConfig(nbits=6, group_size=64)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
    
    
    for i in range(n_blocks):
        quant_config[f'blocks.{i}.attn.qkv'] = q6_config
        quant_config[f'blocks.{i}.attn.proj'] = q6_config
        quant_config[f'blocks.{i}.mlp.fc1'] = q4_config
        quant_config[f'blocks.{i}.mlp.fc2'] = q4_config
        
    return quant_config

# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    quant_config = {}
    n_layers = model.config.num_hidden_layers
    
    for i in range(n_layers):

        if i < 16:
            attn_cfg = BaseQuantizeConfig(nbits=8, group_size=64)
            mlp_cfg  = BaseQuantizeConfig(nbits=8, group_size=64)
        elif i < 18:
            attn_cfg = BaseQuantizeConfig(nbits=8, group_size=64)
            mlp_cfg  = BaseQuantizeConfig(nbits=4, group_size=64)
        else:
            attn_cfg = BaseQuantizeConfig(nbits=4, group_size=64)
            mlp_cfg  = BaseQuantizeConfig(nbits=4, group_size=64)
        # Self-Attention 
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = attn_cfg
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = attn_cfg
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = attn_cfg
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = attn_cfg

        # Feed-forward MLP 
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = mlp_cfg
        quant_config[f'model.layers.{i}.mlp.up_proj'] = mlp_cfg
        quant_config[f'model.layers.{i}.mlp.down_proj'] = mlp_cfg
        
    return quant_config