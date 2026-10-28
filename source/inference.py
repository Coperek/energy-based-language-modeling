import math

import torch
from tqdm import tqdm


def _energy(out: torch.Tensor) -> torch.Tensor:
    return torch.logit(out.float().reshape(()), eps=1e-6)


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
            max_steps=240,
            chain_steps=30,
            min_chain_steps=8,
            patience=6,
            stale_chains=3,
            tol=0.02,
            step_size=0.15,
            noise=0.05,
            verify_top_k=4,
            max_new_tokens=64,
            logs=False,
    ):
        assert max_steps >= 1 and chain_steps >= 1

        text = self.history[-1]['text']  # TODO: not instruction tunning yet
        tokenized, kv, max_new_tokens = self._prefill(text, max_new_tokens, logs)
        generated = []
        steps_log = []

        table = self.model.embedding.weight.detach().float()   # (V, D), shares storage with the model
        table_sq = table.pow(2).sum(-1)                         # (V,), used for nearest-token search
        rms = (table_sq.sum() / table.numel()).sqrt()

        def run_chain(ks, vs, budget):
            """Langevin descent from a random token embedding. Returns (lowest-energy point, steps used)."""
            x = table[torch.randint(len(table), (1,), device=table.device)].clone()   # (1, D)
            best_x, best_e, stale, used = x, float("inf"), 0, 0

            for s in range(min(chain_steps, budget)):
                x = x.detach().requires_grad_(True)
                out, _, _ = self.model.forward_cached_embedding(x.unsqueeze(0), ks, vs)
                e = _energy(out)
                (g,) = torch.autograd.grad(e, x)
                used += 1
                print(e)

                e_val = e.item()
                if e_val < best_e:
                    best_x = x.detach().clone()
                if e_val < best_e - tol:
                    stale = 0
                else:
                    stale += 1
                    if used >= min_chain_steps and stale >= patience:
                        break  # plateau: this chain has stopped improving
                best_e = min(best_e, e_val)

                g = g / (g.pow(2).mean().sqrt() + 1e-8)
                temp = noise * (1 - s / chain_steps)
                x = x.detach() - step_size * rms * g + temp * rms * torch.randn_like(x)

            return best_x, used

        for _ in range(max_new_tokens):
            ks, vs = kv.views()
            best_e, best_tok, best_kv = float("inf"), None, None
            seen, used, stale_count = set(), 0, 0

            while used < max_steps and stale_count < stale_chains:
                with torch.enable_grad():
                    x, n_steps = run_chain(ks, vs, max_steps - used)
                used += n_steps

                near = (table_sq - 2.0 * (table @ x[0])).topk(verify_top_k, largest=False).indices.tolist()

                improved = False
                with torch.no_grad():
                    for tok in near:
                        if tok in seen:
                            continue
                        seen.add(tok)
                        out, new_k, new_v = self.model.forward_cached(tok, ks, vs)
                        e = _energy(out).item()
                        if e < best_e - tol:
                            improved = True
                        if e < best_e:
                            best_e, best_tok, best_kv = e, tok, (new_k, new_v)

                stale_count = 0 if improved else stale_count + 1

            steps_log.append(used)
            if logs:
                p_fake = 1.0 / (1.0 + math.exp(-best_e))
                print(f"{self.tokenizer.decode([best_tok])}: {p_fake} ({used} steps)")

            if best_tok == self.eos_id:
                break

            kv.append(*best_kv)
            generated.append(best_tok)
            tokenized.append(best_tok)

        if logs:
            if steps_log:
                print(f"Gradient steps per token: avg {sum(steps_log) / len(steps_log):.1f}, "
                      f"min {min(steps_log)}, max {max(steps_log)}")
            print("Result:", "".join(self.tokenizer.decode(tokenized)))
        return self.tokenizer.decode(generated)

    @torch.no_grad()
    def generate_exact(self, p=0.01, max_new_tokens=64, logs=False):
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
                for tok, pen in zip(order, penalty):
                    if pen >= best:
                        break

                    out, new_k, new_v = self.model.forward_cached(tok, ks, vs)
                    score = out.item() + pen
                    if score < best:
                        best, best_tok, best_kv = score, tok, (new_k, new_v)

                    pbar.update(1)

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