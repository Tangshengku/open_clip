import os
import copy
import numpy as np
import random
import torch
import math
from torch import nn
from torch.nn import Module
import webdataset as wds
from transformers.modeling_utils import prune_linear_layer
from training.timing import benchmark_foo


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

def param_to_linear(weight, bias):
    inp_dim, out_dim = weight.shape
    layer = nn.Linear(out_dim, inp_dim, device=weight.device)
    layer.weight = weight
    layer.bias = bias
    return layer

def find_all_layer(module):
    res = {}
    for name, param in module.named_parameters():
        if 'attn' in name and name.endswith('weight'):
            name_bias, param_bias = next(iter(module.named_parameters()))
            layer = param_to_linear(param, param_bias)
            res.update({name.rstrip('.weight'): layer})
        if 'mlp' in name and name.endswith('weight'):
            name_bias, param_bias = next(iter(module.named_parameters()))
            layer = param_to_linear(param, param_bias)
            res.update({name.rstrip('.weight'): layer})
    return res



class NoAttention(Module):
    def forward(
        self, hidden_states
    ):
        return hidden_states

class NoAttentionOutput(Module):
    def forward(self, hidden_states):
        return hidden_states

class NoIntermediate(Module):
    def __init__(self, dtype) -> None:
        super().__init__()
        self.weight = torch.empty(1)
        self.weight.to(dtype)
    def forward(self, hidden_states):
        return hidden_states


class NoOutput(Module):
    def forward(self, input_tensor):
        return input_tensor


def shrink(model, update_mask=False, _print=False):
    visual = model
    for i, layer in enumerate(visual.transformer.resblocks):
        if not isinstance(layer.attn, NoAttention):
            weight = layer.attn.out_proj_linear.weight
            if torch.all(weight == 0):
                if update_mask:
                    mask_ = torch.zeros_like(weight.t(), device=weight.device)
                    layer.attn.out_proj_linear.mask = mask_.t()
                    layer.attn.in_proj_linear_q.mask = mask_
                    layer.attn.in_proj_linear_k.mask = mask_
                    layer.attn.in_proj_linear_v.mask = mask_
                else:
                    layer.attn.in_proj_linear_q = NoAttention()
                    layer.attn.in_proj_linear_k = NoAttention()
                    layer.attn.in_proj_linear_v = NoAttention()
                    layer.attn.out_proj_linear = NoAttentionOutput()
            else:
                mask = torch.all(
                    weight.t().reshape((-1, weight.shape[0] * layer.attn.head_size)) == 0, 1
                )
                if update_mask:
                    if _print:
                        print("layer.attn.out_proj_linear.weight: ", i)

                    mask_ = (~mask.unsqueeze(1)) * torch.ones_like(weight.t(), device=weight.device).reshape(
                            (-1, weight.shape[0] * layer.attn.head_size)) 
                    mask_ = mask_.reshape(-1, weight.shape[0])
                    # mask_ = torch.ones_like(mask_, device=mask_.device)
                    layer.attn.out_proj_linear.mask = mask_.t()
                    layer.attn.in_proj_linear_q.mask = mask_
                    layer.attn.in_proj_linear_k.mask = mask_
                    layer.attn.in_proj_linear_v.mask = mask_
                else:
                    idx = []
                    count = 0
                    for i in range(mask.numel()):
                        while count in layer.attn.pruned_heads:
                            count += 1
                        if mask[i]:
                            idx.append(count)
                        count += 1
                    # print("head sparsity: ", len(idx)/layer.attn.num_heads, "  ", layer.attn.num_heads)
                    if torch.any(mask):
                        layer.attn.prune_heads(idx)
        if not isinstance(layer.mlp[2], NoOutput):
            weight = layer.mlp[2].weight
            if torch.all(weight == 0):
                if update_mask:
                    mask_ = torch.zeros_like(weight, device=weight.device)
                    layer.mlp[2].mask = mask_
                    layer.mlp[0].mask = mask_.t()
                else:
                    layer.mlp[0] = NoIntermediate(weight.dtype)
                    layer.mlp[2] = NoOutput()
            else:
                mask = torch.all(weight == 0, 0)
                if update_mask:
                    if _print:
                        print("layer.mlp[0].weight: ",i,)
                        print("layer.mlp[2].weight: ",i,)
                    mask_ = (~mask.unsqueeze(0)) * torch.ones_like(weight, device=weight.device)
                    layer.mlp[2].mask = mask_
                    layer.mlp[0].mask = mask_.t()
                elif torch.any(mask):
                    idx = torch.nonzero(~mask).flatten()
                    # print("fc sparsity: ", 1 - len(idx)/weight.shape[1])
                    layer.mlp[0] = prune_linear_layer(layer.mlp[0], idx)
                    layer.mlp[2] = prune_linear_layer(layer.mlp[2], idx, dim=1)

    if _print:
        for i, layer in enumerate(visual.transformer.resblocks):
            print("layer.attn.out_proj_linear.weight: ", i, '  ', layer.attn.out_proj_linear.weight.shape)
            print("layer.mlp[0].weight: ",i, '  ', layer.mlp[0].weight.shape)
            print("layer.mlp[2].weight: ",i, '  ', layer.mlp[2].weight.shape)

class ZipLM:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        # Accumulate in double precision
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.double)
        self.nsamples = 0

    def add_batch(self, inp, out):
        tmp = inp.shape[1]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.H += 2 / self.nsamples * (inp.matmul(inp.t())).double()

    def invert(self, H, percentdamp=.01):
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except RuntimeError:
            diagmean = torch.mean(torch.diag(H))
            print('Hessian not full rank.')
            tmp = (percentdamp * diagmean) * torch.eye(self.columns, device=self.dev)
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + tmp))
        return Hinv

    def prepare(self):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        H = self.H.float()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        Hinv = self.invert(H)
        Losses = torch.zeros([self.rows, self.columns + 1], device=self.dev)
        return W, H, Hinv, Losses

    def prune_struct(self, pruned, size=1):
        pruned = pruned[:]
        W, H, Hinv, Losses = self.prepare()

        count = self.columns // size
        Losses = torch.zeros(count + 1, device=self.dev)
        mask = torch.zeros(count, device=self.dev).bool()
        rangecount = torch.arange(count, device=self.dev)
        rangecolumns = torch.arange(self.columns, device=self.dev)

        res = []
        if 0 in pruned:
            res.append(self.layer.weight.data.clone())
            pruned = pruned[1:]
            print('   0 error 0.0')
            if not pruned:
                return res
        if size == 1:
            for dropped in range(count + 1):
                diag = torch.diagonal(Hinv)
                scores = torch.sum(W ** 2, 0) / diag
                scores[mask] = float('inf')
                j = torch.argmin(scores)
                Losses[dropped] = scores[j]
                row = Hinv[j, :]
                d = diag[j]
                W -= ((W[:, j] / d).unsqueeze(1)).matmul(row.unsqueeze(0))
                mask[j] = True
                W[:, mask] = 0
                while dropped + 1 == pruned[0]:
                    res.append(W.clone().reshape(self.layer.weight.shape))
                    print('%4d error' % pruned[0], torch.sum(Losses).item() / 2)
                    pruned.pop(0)
                    if not len(pruned):
                        break
                if not len(pruned):
                    break
                row /= torch.sqrt(d)
                Hinv -= row.unsqueeze(1).matmul(row.unsqueeze(0))
        else:
            mask1 = torch.zeros(self.columns, device=self.dev).bool()
            for dropped in range(count + 1):
                blocks = Hinv.reshape(count, size, count, size)
                blocks = blocks[rangecount, :, rangecount, :]
                try:
                    invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
                except:
                    invblocks = torch.linalg.pinv(blocks, hermitian=True)
                W1 = W.reshape((self.rows, count, size)).transpose(0, 1)
                lambd = torch.bmm(W1, invblocks)
                scores = torch.sum(lambd * W1, (1, 2))
                scores[mask] = float('inf')
                j = torch.argmin(scores)
                Losses[dropped] = scores[j]
                rows = Hinv[(size * j):(size * (j + 1)), :]
                d = invblocks[j]
                W -= lambd[j].matmul(rows)
                mask[j] = True
                mask1[(size * j):(size * (j + 1))] = True
                W[:, mask1] = 0
                while dropped + 1 == pruned[0]:
                    res.append(W.clone().reshape(self.layer.weight.shape))
                    print('%4d error' % pruned[0], torch.sum(Losses).item() / 2)
                    pruned.pop(0)
                    if not len(pruned):
                        break
                if not len(pruned):
                    break
                Hinv -= rows.t().matmul(d.matmul(rows))
                Hinv[rangecolumns[mask1], rangecolumns[mask1]] = 1

        return res

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


def gen_transformerdb(
    filename,
    get_model, run, dataloader,
    dataloader_passes=1,
    sparsities=[], min_sparsity=0, max_sparsity=.99, delta_sparse=.1,
    headcount=12, headsize=64, fcdim=4*768,
    attname='attn', fcname='mlp'
):
    modelp = get_model()
    modeld = get_model()
    layersp = find_layers(modelp)
    layersd = find_layers(modeld)

    if not sparsities:
        sparsities = []
        density = 1 - min_sparsity
        while density > 1 - max_sparsity:
            sparsities.append(1 - density)
            density *= 1 - delta_sparse

    ziplm = {}
    for name in layersp:
        layer = layersp[name]
        ziplm[name] = ZipLM(layer)

    def add_batch(name):
        def tmp(layer, inp, out):
            ziplm[name].add_batch(inp[0].data, out.data)
        return tmp
    handles = []
    for name in ziplm:
        handles.append(layersd[name].register_forward_hook(add_batch(name)))
    for i in range(dataloader_passes):
        for j, batch in enumerate(dataloader):
            print(i, j)
            with torch.no_grad():
                run(modeld, batch)
    for h in handles:
        h.remove()

    def prundim(name):
        if attname in name:
            return headsize
        if fcname in name:
            return 1
        return 0

    db = {}
    for name in ziplm:
        print(name)
        size = prundim(name)
        if size > 0:
            print('Structured pruning ...')
            if attname in name:
                sparsities1 = [i / headcount for i in range(headcount)]
                remaining = ziplm[name].layer.weight.shape[1] // headsize
                sparsities1 = sparsities1[(headcount - remaining):]
                pruned = [i for i in range(remaining)]
            else:
                sparsities1 = sparsities
                remaining = ziplm[name].layer.weight.shape[1]
                pruned = [round((1 - s) * fcdim) for s in sparsities1]
                pruned = [remaining - p for p in pruned if p <= remaining]
                sparsities1 = sparsities1[-len(pruned):]
            Ws = ziplm[name].prune_struct(pruned, size=size)
            db[name] = {('%.4f' % s): w.cpu() for s, w in zip(sparsities1, Ws)}
        ziplm[name].free()

    torch.save(db, filename)

class StructuredSPDY:
    def __init__(
        self,
        target,
        db, errors, baselinetime, prunabletime, timings,
        get_model, run, dataloader, loss_fn=None, vision_prune=True,
        dpbuckets=10000,
    ):
        self.target = target
        self.db = db
        self.run = run
        self.dpbuckets = dpbuckets
        self.loss_fn = loss_fn
        self.vision_prune = vision_prune
        self.model = get_model()            
        self.modelp = copy.deepcopy(self.model.visual if self.vision_prune else self.model.text)
        
        self.layersp = find_layers(self.modelp)

        self.batches = []
        for batch in dataloader:
            self.batches.append(run(self.modelp, batch, retmoved=True))

        self.layers = list(db.layers())
        self.sparsities = [list(errors[self.layers[l]].keys()) for l in range(len(self.layers))]
        self.costs = [
            [errors[self.layers[l]][s] for s in self.sparsities[l]] for l in range(len(self.layers))
        ]
        self.timings = [
            [timings[self.layers[l]][s] for s in self.sparsities[l]] for l in range(len(self.layers))
        ]

        self.baselinetime = baselinetime
        self.prunabletime = prunabletime
        if self.baselinetime is None:
            self.baselinetime = self.prunabletime
        targettime = self.baselinetime / self.target - (self.baselinetime - self.prunabletime)
        best = sum(min(c) for c in self.timings)
        if self.prunabletime < self.baselinetime:
            print('Max target:', self.baselinetime / (best + self.baselinetime - self.prunabletime))
        self.bucketsize = targettime / self.dpbuckets

        for row in self.timings:
            for i in range(len(row)):
                row[i] = int(round(row[i] / self.bucketsize))
        
        print('Loss/Base:', self.get_loss(self.model))

    def dp(self, costs):
        DP = np.full((len(costs), self.dpbuckets + 1), float('inf'))
        PD = np.full((len(costs), self.dpbuckets + 1), -1)

        for sparsity in range(len(costs[0])):
            if costs[0][sparsity] < DP[0][self.timings[0][sparsity]]:
                DP[0][self.timings[0][sparsity]] = costs[0][sparsity]
                PD[0][self.timings[0][sparsity]] = sparsity
        for layer in range(1, len(DP)):
            for sparsity in range(len(costs[layer])):
                timing = self.timings[layer][sparsity]
                score = costs[layer][sparsity]
                if timing == 0:
                    tmp = DP[layer - 1] + score
                    better = tmp < DP[layer]
                    if np.sum(better):
                        DP[layer][better] = tmp[better]
                        PD[layer][better] = sparsity
                    continue
                if timing > self.dpbuckets:
                    continue
                tmp = DP[layer - 1][:-timing] + score
                better = tmp < DP[layer][timing:]
                if np.sum(better):
                    DP[layer][timing:][better] = tmp[better]
                    PD[layer][timing:][better] = sparsity

        score = np.min(DP[-1, :])
        timing = np.argmin(DP[-1, :])

        solution = []
        for layer in range(len(DP) - 1, -1, -1):
            solution.append(PD[layer][timing])
            timing -= self.timings[layer][solution[-1]]
        solution.reverse()
        return solution

    def gen_costs(self, coefs):
        return [
            [self.costs[i][j] * coefs[i] for j in range(len(self.costs[i]))] \
            for i in range(len(self.costs))
        ]

    def stitch_model(self, solution):
        model = copy.deepcopy(self.modelp)
        layers = find_layers(model)
        config = {
            self.layers[i]: self.sparsities[i][solution[i]] for i in range(len(self.layers))
        }
        self.db.stitch(layers, config)
        shrink(model)
        return model

    @torch.no_grad()
    def get_loss(self, model):
        loss = 0
        for batch in self.batches:
            loss += self.run(model, batch, loss=True, loss_fn=self.loss_fn)
        return loss / len(self.batches)

    def get_score(self, coefs):
        costs = self.gen_costs(coefs)
        solution = self.dp(costs)
        model = self.stitch_model(solution)
        if self.vision_prune:
            self.model.visual = model
        else:
            self.model.text = model
        return self.get_loss(self.model)

    def save_profile(self, coefs, filename=''):
        solution = self.dp(self.gen_costs(coefs))
        if filename:
            with open(filename, 'w') as f:
                for i in range(len(solution)):
                    f.write('%s %s\n' % (self.sparsities[i][solution[i]], self.layers[i]))
        else:
            for i in range(len(solution)):
                print('%s %s' % (self.sparsities[i][solution[i]], self.layers[i]))

    def score(self, filename):
        with open(filename, 'r') as f:
            solution = []
            i = 0
            for l in f.readlines():
                splits = l.split(' ')
                sparsity = splits[0]
                name = splits[1][:-1]
                while self.layers[i] != name:
                    solution.append(len(self.sparsities[i]) - 1)
                    i += 1
                j = self.sparsities[i].index(sparsity)
                solution.append(j)
                i += 1
        print('Speedup:', self.baselinetime / (
            self.baselinetime - self.prunabletime + \
            sum(t[s] for s, t in zip(solution, self.timings)) * self.bucketsize
        ))
        print('Loss/Pruned:', self.get_loss(self.stitch_model(solution)))

    def dpsolve(self, save=''):
        coefs = np.ones(len(self.layers))
        print('Loss/Pruned:', self.get_score(coefs))
        self.save_profile(coefs)
        if save:
            self.save_profile(coefs, save)

    def search(
        self, save='', randinits=100, searchsteps=1000, muteprob=.1
    ):
        print('Random inits ...')
        candidates = []
        for i in range(randinits):
            coefs = np.zeros(len(self.layers))
            for j in range(len(coefs)):
                coefs[j] = random.random()
            score = self.get_score(coefs)
            candidates.append((score, coefs))
            print('%04d  %.4f %.4f' % (i, min(c[0] for c in candidates), score))
        candidates.sort(key=lambda c: c[0])

        print('Local search ...')
        score, coefs = candidates[0]
        for i in range(searchsteps):
            coefs1 = coefs.copy()
            for j in range(len(coefs)):
                if random.random() < muteprob:
                    coefs1[j] = random.random()
            score1 = self.get_score(coefs1)
            print('%04d  %.4f %.4f' % (i, score, score1))
            if score1 < score:
                score = score1
                coefs = coefs1

        self.save_profile(coefs)
        if save:
            self.save_profile(coefs, save)


class StructDatabase:
    def __init__(self, path, dense):
        self.db = torch.load(path)
        denselayers = find_layers(dense)
        dev = next(iter(denselayers.values())).weight.device
        for name in self.db:
            for sparsity in list(self.db[name].keys()):
                self.db[name][sparsity] = self.db[name][sparsity].to(dev)
        for name in self.db:
            self.db[name]['1.0000'] = torch.zeros_like(denselayers[name].weight.data)
        sd = dense.state_dict()
        self.biases = {n: sd[n + '.bias'] for n in self.db}

    def layers(self):
        return list(self.db.keys())

    def load(self, layers, name, config='0.0000', sd=None):
        if sd is not None:
            layers[name].weight.data = sd[name + '.weight']
            layers[name].bias.data = sd[name + '.bias']
            return
        layers[name].weight.data = self.db[name][config]
        layers[name].bias.data = self.biases[name]
        # if 'attn' in name:
        #     mask = torch.all(
        #                 layers[name].weight.t().reshape((-1, layers[name].weight.shape[0] * layers[name].head_size)) == 0, 1
        #             )
        #     mask_ = (~mask.unsqueeze(1)) * torch.ones_like(layers[name].weight.t(), device=mask.device).reshape(
        #         (-1, layers[name].weight.shape[0] * layers[name].head_size)) 
        #     mask_ = mask_.reshape(-1, layers[name].weight.shape[0])
        #     layers[name].mask = mask_
        # elif 'c_fc' in name:
        #     mask = torch.all(layers[name].weight == 0, 0)
        #     mask_ = (~mask.unsqueeze(1)) * torch.ones_like(layers[name].weight.t(), device=mask.device)

        if config == '1.0000':
            layers[name].bias.data = torch.zeros_like(layers[name].bias.data)

    def stitch(self, layers, config):
        for name in config:
            self.load(layers, name, config[name])

    def load_file(self, model, profile):
        config = {}
        with open(profile, 'r') as f:
            for line in f.readlines():
                splits = line.split(' ')
                sparsity = splits[0]
                name = splits[1][:-1]
                config[name] = sparsity
        for name in self.db:
            if name not in config:
                config[name] = '1.0000'
        layers = find_layers(model)
        self.stitch(layers, config)

    def load_errors(self, path):
        errors = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                name = lines[i].strip()
                errors[name] = {}
                i += 1
                for _ in range(len(self.db[name])):
                    err, level = lines[i].strip().split(' ')
                    errors[name][level] = float(err)
                    i += 1
        return errors

    def get_berttimings(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            baselinetime = float(lines[1])
            prunabletime = float(lines[3])
            i = 5
            attention = {}
            while ' ' in lines[i]:
                time, level = lines[i].strip().split(' ')
                attention[level] = float(time)
                i += 1
            fc = {}
            i += 1
            while i < len(lines):
                time, level = lines[i].strip().split(' ')
                fc[level] = float(time)
                i += 1
        timings = {}
        for name in self.db:
            timings[name] = attention if ('attention' in name or 'attn' in name) else fc
        return baselinetime, prunabletime, timings


def compute_pnorm(p, db, get_model, dataloader, run, filename):
    modeld = get_model()
    modelp = get_model()
    layersd = find_layers(modeld)
    layersp = find_layers(modelp)

    errs = {n: {} for n in db.layers()}
    def accumerrs(name):
        def tmp(layer, inp, out):
            errs[name]['dense'] = errs[name].get('dense', 0) + torch.sum(torch.abs(out.data) ** p).item()
            for config in sorted(db.db[name]):
                db.load(layersp, name, config)
                errs[name][config] = errs[name].get(config, 0) + torch.sum(torch.abs(layersp[name](inp[0].data) - out.data) ** p).item()
        return tmp
    for name in db.layers():
        layersd[name].register_forward_hook(accumerrs(name))

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(i)
            run(modeld, batch)

    with open(filename, 'w') as f:
        for name in errs:
            f.write(name + '\n')
            for config in sorted(errs[name]):
                if config != 'dense':
                    f.write('%.6f %s\n' % (errs[name][config] / errs[name]['dense'], config))

def compute_squared(db, get_model, dataloader, run, filename):
    compute_pnorm(2, db, get_model, dataloader, run, filename)


def _dataloader_builder(dataset, batchsize=256, nsamples=2048):
    default_loader = dataset.dataloader
    template = dict(default_loader.__dict__)

    # drop attributes that will be auto-initialized
    to_drop = [k for k in template if k.startswith("_") or k == "batch_sampler"]
    for item in to_drop:
        template.pop(item)

    # shuffle dataset and select nsamples from it
    # shuffled_dataset = template['dataset'].shuffle(seed=42)
    # nsamples = len(shuffled_dataset) if nsamples == -1 else nsamples
    # shuffled_dataset = shuffled_dataset.select(range(nsamples))
    template["pipeline"][0].dataset.pipeline[-1] = wds.batched(batchsize, partial=False)
    dataset_new = wds.DataPipeline(template["pipeline"][0].dataset.pipeline)
    kwargs = {
        'batch_size': None,
        'dataset': dataset_new,
    }
    template.update(kwargs)
    data_loader = type(default_loader)(**kwargs)

    data_loader.num_batches = math.ceil(nsamples / batchsize)
    data_loader.num_samples = nsamples

    for i, sample in  enumerate(data_loader):
        # sample = dataloader._prepare_inputs(sample)
        if (i + 1) * (batchsize) <= nsamples:
            yield sample
        else:
            break

@torch.no_grad()
def _get_model(module):
    def foo():
        res = copy.deepcopy(module)
        res.eval()
        return res
    return foo

@torch.no_grad()
def _run_clip(model, batch, loss=False, retmoved=False, loss_fn=None):
    dev = next(iter(model.parameters())).device
    image, text = batch
    text = text.to(dev)
    image = image.to(dev)
    # for k, v in batch.items():
    #     batch[k] = v.to(dev)
    if retmoved:
        return batch
    out = model(image, text)
    if loss and loss_fn is not None:
        out_loss = loss_fn(**out, output_dict=True, gather=False)
        return  sum(out_loss.values())
    return out
    # return torch.cat([out[key] for key in ['start_logits', 'end_logits']])

@torch.no_grad()
def _run_clip_vit(model, batch, loss=False, retmoved=False, loss_fn=None):
    dev = next(iter(model.parameters())).device
    image, _ = batch
    image = image.to(dev)
    # for k, v in batch.items():
    #     batch[k] = v.to(dev)
    if retmoved:
        return batch
    out = model(image)
    if loss and loss_fn is not None:
        out_loss = loss_fn(**out, output_dict=True)
        return  sum(out_loss.value())
    return out

@torch.no_grad()
def _run_clip_text(model, batch, loss=False, retmoved=False, loss_fn=None):
    dev = next(iter(model.parameters())).device
    _, text = batch
    text = text.to(dev)
    # for k, v in batch.items():
    #     batch[k] = v.to(dev)
    if retmoved:
        return batch
    out = model(text)
    if loss and loss_fn is not None:
        out_loss = loss_fn(**out, output_dict=True)
        return  sum(out_loss.value())
    return out
    # return torch.cat([out[key] for key in ['start_logits', 'end_logits']])


@torch.no_grad()
def oneshot_prune(args, dataset, module: Module, target: float, loader_batchsize: int, loader_nsamples: int, timings_file: str, loss=None, vision_prune=True):
    if args.rank:
        return
    db_file = f'database_{target}.db'
    if vision_prune:
        model_p = module.visual
    else:
        model_p = module.text
    headcount = model_p.transformer.resblocks[0].attn.num_heads
    headsize = model_p.transformer.width // model_p.transformer.resblocks[0].attn.num_heads
    fcdim = model_p.transformer.resblocks[0].mlp[0].out_features
    db, errors, baselinetime, prunabletime, timings = None, None, None, None, None
    profile = f'profile_{target}.txt'

    gen_transformerdb(
        db_file,
        _get_model(model_p),
        _run_clip_vit if vision_prune else _run_clip_text,
        _dataloader_builder(
            dataset,
            batchsize=loader_batchsize,
            nsamples=loader_nsamples,
        ),
        headcount=headcount,
        headsize=headsize,
        fcdim=fcdim,
        attname='attn.out_proj_linear',
        fcname='mlp.c_proj',
    )

    model = _get_model(model_p)()
    db = StructDatabase(db_file, model)
    error_file = f'errors_squared_{target}.txt'
        
    compute_squared(
        db,
        _get_model(model_p),
        _dataloader_builder(
            dataset,
            batchsize=loader_batchsize,
            nsamples=loader_nsamples,
        ),
        _run_clip_vit if vision_prune else _run_clip_text,
        error_file
    )

    errors = db.load_errors(error_file)
    baselinetime, prunabletime, timings = db.get_berttimings(timings_file)


    struct_spdy = StructuredSPDY(
        target, db, errors, baselinetime, prunabletime, timings,
        _get_model(module), _run_clip,
        _dataloader_builder(
            dataset,
            batchsize=loader_batchsize,
            nsamples=loader_nsamples,
        ),
        loss_fn=loss,
        vision_prune=vision_prune
    )

    struct_spdy.search(profile)
    # db.load_file(model_p, profile)
    # shrink(module.visual, update_mask=True, _print=True)


    # test_Dataloader = _dataloader_builder(
    #         dataset,
    #         batchsize=loader_batchsize,
    #         nsamples=loader_nsamples,
    #     )
    # for batch in test_Dataloader:
    #     test_image = batch[0].to(next(iter(module.parameters())).device)
    #     res, _ = module.visual(test_image)
    #     module.visual.mask_weights()
    #     print(res.equal(module.visual(test_image)[0]))

    # os.remove(db_file)
    test_speed = False
    if test_speed:
        dev = next(iter(module.parameters())).device
        image_inputs, text_inputs = next(iter(dataset.dataloader))
        image_inputs = image_inputs.to(dev)
        mean, std = benchmark_foo(lambda: module.visual(image_inputs))
        print("pruned speed: ", mean/1000)

@torch.no_grad()
def load_pruned_model(module, db_file, profile, vision_prune):
    # db_file = f'laion2b_e16/database_{target}_10000.db'
    if vision_prune:
        model_p = module.visual
    else:
        model_p = module.text
    
    # profile = f'laion2b_e16/profile_{target}_10000.txt'

    model = _get_model(model_p)()
    db = StructDatabase(db_file, model)
    
    db.load_file(model_p, profile)
    shrink(module.visual, update_mask=False, _print=True)

def test_speed(module, dataset):
    dev = next(iter(module.parameters())).device
    image_inputs, text_inputs = next(iter(dataset.dataloader))
    image_inputs = image_inputs.to(dev)
    mean, std = benchmark_foo(lambda: module.visual(image_inputs))
    print("pruned speed: ", mean/1000)