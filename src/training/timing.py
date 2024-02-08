import torch
from copy import deepcopy
from transformers.modeling_utils import prune_linear_layer, prune_conv1d_layer


@torch.no_grad()
def benchmark_foo(foo, repetitions=100, with_attn_cache=False, attn_cache_kwargs=None):
    # helpers
    timings = torch.zeros(repetitions)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.backends.cudnn.benchmark = True

    if with_attn_cache:
        batchsize = attn_cache_kwargs['batchsize']
        num_heads = attn_cache_kwargs['num_heads']
        head_size = attn_cache_kwargs['head_size']
        seq_length = attn_cache_kwargs['seq_length']
        # print(f"cache details: {attn_cache_kwargs}")
        key = torch.randn((batchsize, num_heads, seq_length, head_size)).to("cuda")
        value = torch.randn((batchsize, num_heads, seq_length, head_size)).to("cuda")
        layer_past = (key, value)
    # warm-up
    for _ in range(50):
        if with_attn_cache:
            foo(layer_past)
        else:
            foo()

    # actual timing loop
    torch.cuda.synchronize()
    for i in range(repetitions):
        torch.cuda.synchronize()
        starter.record()
        if with_attn_cache:
            foo(layer_past)
        else:
            foo()
        torch.cuda.synchronize()
        ender.record()
        torch.cuda.synchronize()
        timed = starter.elapsed_time(ender)
        timings[i] = timed

    return (timings.mean(), timings.std())


@torch.no_grad()
def timing_main(model, device, data_loader, is_bert=True, debug=False, with_attn_cache=False, repetitions=50): # either BERT or GPT2
    # assume model and trainer are initialized from the main script
    model.to(device)
    model.eval()
    sample_inputs, text_inputs = next(iter(data_loader))
    sample_inputs = sample_inputs.to(device)
    # sample_inputs = {k: v.to(device) for k, v in sample_inputs.items()}
    # for k, v in sample_inputs.items():
    #     print(f"{k} = {v.shape}")

    image_base, attn_layer, prunable, layer = None, None, None, None
    if is_bert:
        # attn_layer = model.bert.encoder.layer[0].attention
        # prunable = model.bert.encoder
        # layer = model.bert.encoder.layer[0]
        image_base = model.visual
        attn_layer = model.visual.transformer.resblocks[0].attn
        prunable = model.visual.transformer
        layer = model.visual.transformer.resblocks[0]
    else:  # is_gpt2 :-)
        attn_layer = model.transformer.h[0].attn
        prunable = model.transformer.h
        layer = model.transformer.h[0]

    db_downscale = 1000  # 1000 for normal, I think 100 for text-gen

    # === cache inputs/outputs needed for timing ===
    cached_inputs_outputs = {}
    def cache_inputs_factory(cache_dict, dict_key):
        def foo(layer, inp, out):
            cache_dict[dict_key + "_inputs"] = inp
            cache_dict[dict_key + "_outputs"] = out
        return foo

    attn_dict_key = "attention"
    attn_hook = attn_layer.register_forward_hook(cache_inputs_factory(cached_inputs_outputs, attn_dict_key))
    prunable_dict_key = "prunable"
    prunable_hook = None
    if is_bert:
        prunable_hook = prunable.register_forward_hook(cache_inputs_factory(cached_inputs_outputs, prunable_dict_key))
    else:
        prunable_hook = prunable[0].register_forward_hook(cache_inputs_factory(cached_inputs_outputs, prunable_dict_key))
    with torch.no_grad():
        _ = image_base(sample_inputs)
    attn_hook.remove()
    prunable_hook.remove()
    assert attn_dict_key + "_inputs" in cached_inputs_outputs.keys()
    assert attn_dict_key + "_outputs" in cached_inputs_outputs.keys()
    assert prunable_dict_key + "_inputs" in cached_inputs_outputs.keys()
    assert prunable_dict_key + "_outputs" in cached_inputs_outputs.keys()

    # === benchmark entire model ===
    print("base")
    t_mean, t_std = benchmark_foo(lambda: image_base(sample_inputs), repetitions=repetitions)
    print(f"{t_mean/db_downscale:.4f}")

    # === benchmark prunable parts ===
    print("prunable")
    if is_bert:
        t_mean, t_std = benchmark_foo(lambda: prunable(*cached_inputs_outputs[prunable_dict_key + "_inputs"]), repetitions=repetitions)
    else: # is_gpt2
        # prunable is ModuleList
        def fw_modulelist(mlist, x):
            def run():
                nonlocal x
                for m in mlist:
                    x = m(*x)
            return run
        t_mean, t_std = benchmark_foo(fw_modulelist(prunable, cached_inputs_outputs[prunable_dict_key + "_inputs"]), repetitions=repetitions)
    print(f"{t_mean/db_downscale:.4f}")

    # === lm-head gpt2 ===
    if not is_bert:
        print("lm_head")
        t_mean, t_std = benchmark_foo(lambda: model.lm_head(*cached_inputs_outputs[prunable_dict_key + "_outputs"][0]), repetitions=repetitions)
        print(f"{t_mean/db_downscale:.4f}")

    # === benchmark attention layer ===
    num_heads = 12
    # num_heads = model.config.num_attention_heads if is_bert else model.config.n_head
    print(f"[INFO] This model has num_heads={num_heads}")
    print("attention")
    for pruned_heads_idx in range(0, num_heads):
        pruned_attn = deepcopy(attn_layer)
        pruned_attn.prune_heads(list(range(pruned_heads_idx)))

        if with_attn_cache:
            _batchsize = sample_inputs['input_ids'].shape[0]
            _num_heads = model.config.n_head - pruned_heads_idx
            _head_size = model.config.n_embd // model.config.n_head
            _seq_length = 50
            t_mean, t_std = benchmark_foo(lambda x: pruned_attn(*cached_inputs_outputs[attn_dict_key + "_inputs"], layer_past=x), repetitions=repetitions, with_attn_cache=True, attn_cache_kwargs={'batchsize': _batchsize, 'num_heads': _num_heads, 'head_size': _head_size, 'seq_length': _seq_length})
        else:
            t_mean, t_std = benchmark_foo(lambda : pruned_attn(*cached_inputs_outputs[attn_dict_key + "_inputs"]), repetitions=repetitions)
        if debug:
            print(f"[DEBUG] pruned_attn.shape = {pruned_attn.c_proj.weight.shape}")
        print(f"{t_mean/db_downscale:.4f} {pruned_heads_idx/num_heads:.4f}")
    print(f"0.0000 1.0000") # all pruned have time=0

    # === benchmark ffn layer ===
    inter_size = None
    if is_bert:
        inter_size = 3072
        # inter_size = model.config.intermediate_size if hasattr(model.config, 'intermediate_size') else model.config.hidden_size * 4
    else:  # is_gpt2
        inter_size = model.config.n_inner if hasattr(model.config, 'n_inner') and model.config.n_inner is not None else model.config.n_embd * 4
    print(f"[INFO] This model has intermediate_size={inter_size}")
    print("fc")
    delta = .9
    sparsities = [1 - delta ** i for i in range(100) if delta ** i > .01]
    for sparsity in sparsities:
        pruned_inter_dim = inter_size - round(inter_size * sparsity)
        idx = torch.arange(pruned_inter_dim)

        if is_bert:
            layer.mlp[0] = prune_linear_layer(layer.mlp[0], idx)
            layer.mlp[2] = prune_linear_layer(layer.mlp[2], idx, dim=1)
            t_mean, t_std = benchmark_foo(lambda: layer(cached_inputs_outputs[attn_dict_key + "_outputs"][0]), repetitions=repetitions)
        else: # is_gpt2
            layer.mlp.c_fc = prune_conv1d_layer(layer.mlp.c_fc, idx, dim=1)
            layer.mlp.c_proj = prune_conv1d_layer(layer.mlp.c_proj, idx, dim=0)
            t_mean, t_std = benchmark_foo(lambda: layer.mlp(cached_inputs_outputs[attn_dict_key + "_outputs"][0]), repetitions=repetitions)

        if debug:
            if is_bert:
                print(f"[DEBUG] layer.intermediate.dense.weight.shape = {layer.intermediate.dense.weight.shape}")
                print(f"[DEBUG] layer.output.dense.weight.shape = {layer.output.dense.weight.shape}")
            else: # is_gpt2
                print(f"[DEBUG] layer.mlp.c_fc.weight.shape = {layer.mlp.c_fc.weight.shape}")
                print(f"[DEBUG] layer.mlp.c_proj.weight.shape = {layer.mlp.c_proj.weight.shape}")
        print(f"{t_mean/db_downscale:.4f} {sparsity:.4f}")
    print(f"0.0000 1.0000") # all pruned have time=0

