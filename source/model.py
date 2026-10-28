import math
from pathlib import Path

import matplotlib.pyplot as plt

from util.attention import FlashSelfAttention

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
import torch

from util import dataset_interface as data

from tqdm import tqdm

import os


def generate_unigram(indices_windows_iterator, indices_count, max_windows=1_000_000):
    counts = torch.zeros((indices_count,), dtype=torch.long)

    with tqdm(total=max_windows) as pbar:
        for i, indices in enumerate(indices_windows_iterator):
            if i >= max_windows:
                break
            counts += torch.bincount(torch.as_tensor(indices), minlength=indices_count)
            pbar.update(1)

    unigram = counts.float() / counts.sum()
    return unigram


class RMSNorm(nn.Module): # better dtype handling
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # same name and shape as nn.RMSNorm
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (out * self.weight).to(x.dtype)


class EnergyTransformerBlock(nn.Module):

    def __init__(self, dim, heads, rope_theta, context_window=-1, dropout=0.0, max_seq_len=2048):
        super().__init__()

        self.activation = nn.GELU(approximate="tanh")

        self.norm1 = RMSNorm(dim)
        self.attention = FlashSelfAttention(dim, heads, rope_theta, window_size=context_window, max_seq_len=max_seq_len)
        self.dropout1 = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.norm2 = RMSNorm(dim)
        self.linear_up = nn.Linear(dim, dim * 4)
        self.linear_down = nn.Linear(dim * 4, dim)
        self.dropout2 = nn.Dropout(p=dropout / 2.0) if dropout > 0 else nn.Identity()

    def forward_training(self, positives: torch.Tensor, negatives: torch.Tensor):

        positives_norm = self.norm1(positives)
        negatives_norm = self.norm1(negatives)

        k_cache, v_cache = self.attention.calculate_key_value(positives_norm)

        positives_query = self.attention.calculate_query(positives_norm)
        negatives_query = self.attention.calculate_query(negatives_norm)

        k_shifted = torch.cat([torch.zeros_like(k_cache[:, :1, :, :]), k_cache[:, :-1, :, :]], dim=1)
        v_shifted = torch.cat([torch.zeros_like(v_cache[:, :1, :, :]), v_cache[:, :-1, :, :]], dim=1)

        positives_attn = self.dropout1(self.attention.forward_qkv(positives_query, k_shifted, v_shifted))
        negatives_attn = self.dropout1(self.attention.forward_qkv(negatives_query, k_shifted, v_shifted))

        positives_attn = positives_attn + positives
        negatives_attn = negatives_attn + negatives

        mlp_hidden_positives = self.linear_up(self.norm2(positives_attn))
        mlp_hidden_negatives = self.linear_up(self.norm2(negatives_attn))

        positives_out = positives_attn + self.dropout2(self.linear_down(self.activation(mlp_hidden_positives)))
        negatives_out = negatives_attn + self.dropout2(self.linear_down(self.activation(mlp_hidden_negatives)))

        return positives_out, negatives_out

    def forward_cached(self, x, cache_k, cache_v):
        if self.training:
            raise Exception("forward_training() must be used during training instead of forward()")

        pos = cache_k.shape[1] - 1
        attention_in = self.norm1(x)

        my_k, my_v = self.attention.calculate_key_value(attention_in, offset=pos)
        my_q = self.attention.calculate_query(attention_in, offset=pos)

        attention_out = self.attention.forward_qkv(my_q, cache_k, cache_v)

        mlp_in = attention_out + x
        mlp_out = self.linear_down(self.activation(self.linear_up(self.norm2(mlp_in)))) + mlp_in
        return mlp_out, my_k, my_v

    def forward_and_return_cache(self, x: torch.Tensor):
        if self.training:
            raise Exception("forward_training() must be used during training instead of forward()")

        attention_in = self.norm1(x)

        k_cache, v_cache = self.attention.calculate_key_value(attention_in)
        k_cache = torch.cat([torch.zeros_like(k_cache[:, :1, :, :]), k_cache], dim=1)
        v_cache = torch.cat([torch.zeros_like(v_cache[:, :1, :, :]), v_cache], dim=1)
        k_shifted = k_cache[:, :-1, :, :]
        v_shifted = v_cache[:, :-1, :, :]
        query = self.attention.calculate_query(attention_in)

        attention_out = self.attention.forward_qkv(query, k_shifted, v_shifted)
        attention_out = attention_out + x

        mlp_hidden = self.linear_up(self.norm2(attention_out))
        mlp_hidden = self.linear_down(self.activation(mlp_hidden))
        mlp_hidden = mlp_hidden + attention_out

        return mlp_hidden, k_cache, v_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, cache_k, cache_v = self.forward_and_return_cache(x)
        return out


class EnergyTransformer(nn.Module):

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(
            self,
            device,
            dim: int,
            num_heads: int,
            num_blocks: int,
            vocab_size: int,
            max_sequence_length: int,
            rope_theta: int = 10_000,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.unigram = None
        self.device = device
        self.sequence_length = max_sequence_length

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.out = nn.Linear(dim, 1)

        self.blocks = nn.ModuleList([
            EnergyTransformerBlock(dim, num_heads, rope_theta, dropout=dropout, max_seq_len=max_sequence_length)
            for _ in range(num_blocks)
        ])

        for b in self.blocks:
            nn.init.normal_(b.attention.out.weight, std=0.02 / math.sqrt(2 * num_blocks))
            nn.init.normal_(b.linear_down.weight, std=0.02 / math.sqrt(2 * num_blocks))

        self.final_norm = RMSNorm(dim)

        self.to(device)

        print(f"EnergyTransformer initialized with {self.count_parameters()} parameters")

        self.representation = (
            "EnergyTransformer: "
            f"Sequence length: {self.sequence_length}, "
            f"Vocab size: {self.vocab_size}, "
            f"Dim: {dim}, "
            f"Attention heads: {num_heads}, "
            f"Dropout: {dropout}, "
            f"Rope theta: {rope_theta}, "
            f"Block count: {num_blocks}"
        )

    def __repr__(self):
        return self.representation

    @staticmethod
    def loss_func_raw(real_logits, fake_logits, fake_target, reduction='none'):
        real_logits, fake_logits = real_logits.float(), fake_logits.float()

        real_target = torch.zeros_like(real_logits)
        real_loss = F.binary_cross_entropy_with_logits(real_logits, real_target, reduction=reduction)

        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_target.unsqueeze(-1), reduction=reduction)

        return real_loss, fake_loss

    @staticmethod
    def loss_func(real_logits, fake_logits, fake_target, reduction='mean'):
        real_loss, fake_loss = EnergyTransformer.loss_func_raw(
            real_logits, fake_logits, fake_target, reduction=reduction
        )
        return real_loss + fake_loss

    def make_negatives(self, x, aug_mix):
        B, S = x.shape
        dev = x.device
        pos = torch.arange(S, device=dev).expand(B, S)

        same_seq = x.gather(1, torch.randint(0, S, (B, S), device=dev))
        off = torch.randint(1, 9, (B, S), device=dev) * (2 * torch.randint(0, 2, (B, S), device=dev) - 1)
        nearby = x.gather(1, (pos + off).clamp(0, S - 1))
        uni = torch.multinomial(self.unigram, x.numel(), replacement=True).view(B, S)

        src = torch.multinomial(torch.tensor(aug_mix, device=dev), x.numel(), replacement=True).view(B, S)
        neg = torch.where(src == 0, same_seq, torch.where(src == 1, nearby, uni))

        neg_target = (neg != x).float().detach()

        return neg, src, neg_target

    def _blocks_forward_training(self, out1, out2):
        for block in self.blocks:
            out1, out2 = checkpoint(block.forward_training, out1, out2, use_reentrant=False)
        return out1, out2

    def forward_training(self, x: torch.Tensor, aug_mix):
        assert self.unigram is not None

        positives = x
        negatives, source, negatives_target = self.make_negatives(x, aug_mix)

        out1 = self.embedding(positives).to(torch.bfloat16)
        out2 = self.embedding(negatives).to(torch.bfloat16)

        out1, out2 = self._blocks_forward_training(out1, out2)

        out1 = self.out(self.final_norm(out1))
        out2 = self.out(self.final_norm(out2))

        return out1, out2, source, negatives_target

    def forward_cached(self, token_id, k_cache, v_cache):
        emb = self.embedding(torch.tensor([[token_id]], device=self.device))
        return self.forward_cached_embedding(emb, k_cache, v_cache)

    def forward_cached_embedding(self, emb: torch.Tensor, cache_k: list[torch.Tensor], cache_v: list[torch.Tensor]):
        if self.training:
            raise Exception("forward_training() must be used during training instead of forward()")

        my_k_cache, my_v_cache = [], []
        with torch.amp.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16):
            out = emb.to(torch.bfloat16)

            for i, block in enumerate(self.blocks):
                out, new_k_cache, new_v_kache = block.forward_cached(out, cache_k[i], cache_v[i])
                my_k_cache.append(new_k_cache)
                my_v_cache.append(new_v_kache)

            out = F.sigmoid(self.out(self.final_norm(out)).float())
            out = out.squeeze(0).squeeze(0)
            return out, my_k_cache, my_v_cache

    def forward(self, x: torch.Tensor, output_cache=False):
        if self.training:
            raise Exception("forward_training() must be used during training instead of forward()")

        my_k_cache, my_v_cache = [], []
        with torch.amp.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16):
            out = self.embedding(x.to(self.device)).to(torch.bfloat16)

            for block in self.blocks:
                if output_cache:
                    out, k_cache, v_cache = block.forward_and_return_cache(out)
                    my_k_cache.append(k_cache)
                    my_v_cache.append(v_cache)
                else:
                    out = block.forward(out)

            out = F.sigmoid(self.out(self.final_norm(out)).float())

        if output_cache:
            return out, my_k_cache, my_v_cache

        return out

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        c = torch.load(path, weights_only=False)
        self.load_state_dict(c["model"])
        self.unigram = c["unigram"]

    def save_checkpoint(
            self,
            path,
            optimizer,
            scheduler,
            progress,
            loss_history
    ):
        tmp_path = f"{path}.tmp"

        try:
            with open(tmp_path, "wb") as f:
                torch.save({
                    'model': self.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'progress': progress,
                    'loss_history': loss_history,
                    'unigram': self.unigram,
                }, f)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def evaluate_model(self, batch_iterator, aug_mix):
        self.eval()

        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(batch_iterator):
                with torch.amp.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16):
                    out1, out2, source, neg_target = self.forward_training(
                        torch.tensor(batch, device=self.device), aug_mix
                    )
                    total_loss += self.loss_func(out1, out2, neg_target)

        average_loss = total_loss / (i + 1)
        return average_loss

    def fit(
            self,
            train_path: str,
            eval_path: str | None,
            batch_size: int,
            model_save_path: str,
            load_if_exists=True,
            max_tokens: int | None = None,
            compile_blocks: bool = True,
            # hyperparams
            augmentation_methods_mix=(.4, .2, .4),
            max_lr: float=6e-4,
            warmup_steps: float=800,
            start_lr_div_factor=20.0,
            end_lr_div_factor=10.0,
            # hyperparams
            adam_betas=(0.9, 0.95),
            weight_decay: float = 0.1,
            # logging
            logs=True,
            loss_history_delay: int = 5,
            checkpoint_delay: int = 100,
    ):
        assert abs(sum(augmentation_methods_mix) - 1.0) < 1e-6

        if logs:
            print(self.__repr__())

        assert batch_size > 0

        save_path_obj = Path(model_save_path)

        checkpoint_loaded = None
        current_progress = 0

        loss_history = {
            "overall": [],
            "real": [],
            "sequence": [],
            "nearby": [],
            "unigram": [],
        }

        if save_path_obj.exists() and load_if_exists:
            checkpoint_loaded = torch.load(model_save_path, weights_only=False, map_location=self.device)

            self.load_state_dict(checkpoint_loaded['model'])
            current_progress = checkpoint_loaded['progress']
            loss_history = checkpoint_loaded['loss_history']
            if 'unigram' in checkpoint_loaded:
                self.unigram = checkpoint_loaded['unigram']

            if logs:
                plt.plot(loss_history["overall"], label="overall", color="black")
                plt.plot(loss_history["sequence"], label="sequence", linestyle="dashed", color="grey")
                plt.plot(loss_history["nearby"], label="nearby", linestyle="dashed", color="green")
                plt.plot(loss_history["unigram"], label="unigram", linestyle="dashed", color="blue")
                plt.plot(loss_history["real"], label="real", linestyle="dashed", color="orange")
                plt.show()

        if self.unigram is None:
            print("Generating unigram...")
            self.unigram = generate_unigram(
                data.ParquetReader(
                    train_path,
                    config=data.ReaderConfig(output_mode="python", ),
                    columns=0,
                ).read(),
                self.vocab_size
            ).to(self.device)

        if compile_blocks and not getattr(self, "_compiled", False):
            self._blocks_forward_training = torch.compile(self._blocks_forward_training)
            self._compiled = True

        train_reader = data.ParquetReader(
            train_path,
            config=data.ReaderConfig(
                skip_first_n_items=current_progress,
                output_mode="python",
            ),
            columns=0,
        )

        decay = [p for p in self.parameters() if p.ndim >= 2]
        no_decay = [p for p in self.parameters() if p.ndim < 2]

        initial_steps = current_progress // batch_size
        total_steps = (len(train_reader) + current_progress) // batch_size
        if max_tokens is not None:
            total_steps = min(total_steps, round(max_tokens / self.sequence_length / batch_size))

        optimizer = optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=max_lr, betas=adam_betas, fused=True)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=warmup_steps/total_steps,
            anneal_strategy="cos", div_factor=start_lr_div_factor,
            final_div_factor=end_lr_div_factor/start_lr_div_factor, cycle_momentum=False)

        if checkpoint_loaded is not None:
            optimizer.load_state_dict(checkpoint_loaded['optimizer'])
            scheduler.load_state_dict(checkpoint_loaded['scheduler'])

        if logs:
            print(f"Training... (starting from window {current_progress})")
            print(f"Estimated step count: {total_steps}, Spent on warmup: {warmup_steps:.0f}")

        self.train()
        running_loss = 0.0

        with (tqdm(
            initial=initial_steps,
            total=total_steps,
            unit="steps",
        ) as pbar):

            for i, batch in enumerate(
                    train_reader.read_batched(batch_size=batch_size, drop_last=True)
            ):
                if initial_steps + i >= total_steps:
                    break

                optimizer.zero_grad()

                with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                    out1, out2, src, neg_target = self.forward_training(
                        torch.tensor(batch, device=self.device), augmentation_methods_mix
                    )
                    loss_real, loss_fake = self.loss_func_raw(out1, out2, neg_target, reduction="none")

                mean_loss = loss_real.mean() + loss_fake.mean()
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                pbar.update(1)
                estimated_progress = len(batch)
                current_progress += estimated_progress

                loss_val = mean_loss.item()
                running_loss += loss_val

                if (i + 1) % loss_history_delay == 0:
                    loss_fake = loss_fake.detach().squeeze(-1)
                    loss_real = loss_real.detach().squeeze(-1)

                    sequence_loss = torch.where(src == 0, loss_fake, 0.0)
                    loss_history["sequence"].append(
                        (sequence_loss.sum() / ((src == 0).sum() + 1e-6)).item()
                    )

                    nearby_loss = torch.where(src == 1, loss_fake, 0.0)
                    loss_history["nearby"].append(
                        (nearby_loss.sum() / ((src == 1).sum() + 1e-6)).item()
                    )

                    unigram_loss = torch.where(src == 2, loss_fake, 0.0)
                    loss_history["unigram"].append(
                        (unigram_loss.sum() / ((src == 2).sum() + 1e-6)).item()
                    )

                    loss_history["real"].append(loss_real.mean().item())
                    loss_history["overall"].append(loss_val)

                if (i + 1) % checkpoint_delay == 0:
                    if logs:
                        print(f"Checkpointed at step {i} | Avg Loss: {(running_loss / checkpoint_delay):.4f} | LR: {scheduler.get_last_lr()}")
                        if len(loss_history["overall"]):
                            print(
                                "Last loss: ",
                                f" sequence: {loss_history["sequence"][-1]:.4f}",
                                f" nearby: {loss_history["nearby"][-1]:.4f}",
                                f" unigram: {loss_history["unigram"][-1]:.4f}",
                                f" real: {loss_history["real"][-1]:.4f}",
                                f" overall: {loss_history["overall"][-1]:.4f}",
                            )
                    running_loss = 0.0

                    self.save_checkpoint(model_save_path, optimizer, scheduler, current_progress, loss_history)

        self.save_checkpoint(model_save_path, optimizer, scheduler, current_progress, loss_history)

        if eval_path is not None:
            eval_reader = data.ParquetReader(
                eval_path,
                config=data.ReaderConfig(output_mode="python"),
                columns=0,
            )

            eval_loss = self.evaluate_model(
                eval_reader.read_batched(batch_size=batch_size, drop_last=True),
                augmentation_methods_mix
            )
            print("Evaluation loss for this epoch: ", eval_loss)
