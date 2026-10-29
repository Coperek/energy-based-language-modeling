import random

import torch
from tqdm import tqdm

from itertools import batched


class _KVBuffer:
    """
    Preallocated per-layer K/V cache.
    """
    def __init__(self, ks, vs, extra_slots: int):
        layers, n = len(ks), ks[0].shape[1]
        shape = (layers, 1, n + extra_slots, *ks[0].shape[2:])
        self.k = torch.empty(shape, dtype=ks[0].dtype, device=ks[0].device)
        self.v = torch.empty(shape, dtype=vs[0].dtype, device=vs[0].device)
        for j in range(layers):
            self.k[j, :, :n].copy_(ks[j])
            self.v[j, :, :n].copy_(vs[j])
        self.n = n

    def views(self):
        return (
            [self.k[j, :, :self.n] for j in range(len(self.k))],
            [self.v[j, :, :self.n] for j in range(len(self.v))],
        )

    def append(self, new_k, new_v):
        for j in range(len(self.k)):
            self.k[j, :, self.n:self.n + 1].copy_(new_k[j])
            self.v[j, :, self.n:self.n + 1].copy_(new_v[j])
        self.n += 1


class Session:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.special_tokens["<eos>"]

        self.history = []
        self.restart()

    def restart(self):
        self.history.clear()
        self.model.eval()

    def message(self, text: str, role: str):
        self.history.append({'role': role, 'text': text})

    def _prefill(self, text: str, max_new_tokens: int, logs: bool):
        tokenized = self.tokenizer.encode(text)

        room = self.model.sequence_length - len(tokenized)
        if room <= 0:
            raise ValueError(
                f"prompt has {len(tokenized)} tokens, the model's context is {self.model.sequence_length}"
            )
        max_new_tokens = min(max_new_tokens, room)

        with torch.no_grad():
            output, ks, vs = self.model.forward(torch.tensor(tokenized).unsqueeze(0), output_cache=True)
            if logs:
                for tok, score in zip(tokenized, output.flatten().tolist()):
                    print(f"{self.tokenizer.decode([tok])}: {score}")
            kv = _KVBuffer(ks, vs, max_new_tokens)

        del output, ks, vs
        return tokenized, kv, max_new_tokens

    def generate_langevin(
            self,
            max_steps=32,
            trial_count=64,
            max_output_tokens=100,
            noise=0.1,
            logs=False,
    ):
        assert max_steps >= 1 and trial_count >= 1

        text = self.history[-1]['text']  # TODO: no instruction tunning yet
        tokenized, kv, max_new_tokens = self._prefill(text, max_output_tokens, logs)
        generated = []

        table = self.model.embedding.weight.detach().float()

        for _ in range(max_new_tokens):

            ks, vs = kv.views()
            best_token, best_e, best_kv = 0, float("inf"), None

            visited = torch.full((len(table),), False, device=table.device, dtype=torch.bool)

            for trial in range(trial_count):

                token = random.randint(0, len(table) - 1)

                for i in range(max_steps):

                    emb = table[token].clone()
                    emb = emb.detach().unsqueeze(0).requires_grad_(True)
                    out, new_k, new_v = self.model.forward_cached_embedding(emb.unsqueeze(0), ks, vs)
                    visited[token] = True

                    energy = out.float().reshape(())
                    (grad,) = torch.autograd.grad(energy, emb)
                    grad += torch.randn_like(grad) * noise * torch.var(grad)

                    if energy < best_e:
                        best_e, best_token, best_kv = energy, token, (new_k, new_v)
                        print(f"{self.tokenizer.decode([best_token])}: {best_e}")

                    table_relative = table - emb
                    table_ranked = torch.cosine_similarity(table_relative, grad, dim=-1)
                    table_ranked = torch.where(visited, -1.0, table_ranked)

                    max_index = torch.argmax(table_ranked, dim=-1)

                    if len(max_index.shape) > 0:
                        token = [i.item() for i in max_index][0]
                    else:
                        token = max_index.item()

            if logs:
                print(f"Best continuation: {self.tokenizer.decode([best_token])}")

            if best_token == self.eos_id:
                break

            kv.append(*best_kv)
            generated.append(best_token)
            tokenized.append(best_token)

        return self.tokenizer.decode(generated)

    @torch.no_grad()
    def generate_exact(
            self,
            p=0.01,
            batch_size=1024,
            max_new_tokens=64,
            logs=False
    ):
        text = self.history[-1]['text']  # TODO: not instruction tunning yet
        tokenized, kv, max_new_tokens = self._prefill(text, max_new_tokens, logs)
        generated = []

        unigram = self.model.unigram
        assert unigram is not None, "model.unigram must be set before generate_exact"

        order = torch.argsort(unigram, descending=True)
        penalty = (-torch.log(unigram[order] + 1e-12) * p).tolist()
        order = order.tolist()

        for _ in range(max_new_tokens):
            ks, vs = kv.views()
            best, best_tok, best_kv = float("inf"), None, None

            with tqdm(total=len(order), disable=not logs) as pbar:
                for token_batch, penalty_batch in zip(batched(order, batch_size), batched(penalty, batch_size)):
                    if penalty_batch[0] >= best:
                        break

                    out, new_k, new_v = self.model.forward_cached(torch.tensor(token_batch, device=self.model.device), ks, vs)

                    penalty_batch = torch.tensor(penalty_batch, device=out.device).unsqueeze(-1).unsqueeze(-1)
                    out += penalty_batch

                    batch_best_i = torch.argmin(out)
                    if len(batch_best_i.shape) > 0:
                        batch_best_i = batch_best_i[0].item()
                    else:
                        batch_best_i = batch_best_i.item()
                    batch_best = out[batch_best_i].item()

                    if batch_best < best:
                        batch_best_kv = ([k[batch_best_i] for k in new_k], [v[batch_best_i] for v in new_v])
                        best, best_tok, best_kv = batch_best, token_batch[batch_best_i], batch_best_kv

                    pbar.update(batch_size)

            if logs:
                print(f"Best continuation: {self.tokenizer.decode([best_tok])}")

            if best_tok == self.eos_id:
                break

            kv.append(*best_kv)
            generated.append(best_tok)
            tokenized.append(best_tok)

        if logs:
            print(f"Result: {self.tokenizer.decode(tokenized)}")

        return self.tokenizer.decode(generated)