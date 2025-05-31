import torch
from typing import Optional, Tuple
import torch.nn as nn
import types
from dllm_cache.cache import dLLMCache



def logout_cache_LLaDA(model: nn.Module, tf_block_module_key_name: str) -> None:
    """Restore original functions for transformer blocks, attention, and rotary embeddings."""
    target_module: Optional[nn.ModuleList] = None
    for name, module in model.named_modules():
        if name == tf_block_module_key_name:
            target_module = module
            break
    if target_module is None:
        return
    for tf_block in target_module:
        if hasattr(tf_block, "_old_forward"):
            tf_block.forward = tf_block._old_forward
            delattr(tf_block, "_old_forward")
        if hasattr(tf_block, "_old_attention"):
            tf_block.attention = tf_block._old_attention
            delattr(tf_block, "_old_attention")
        if hasattr(tf_block.rotary_emb, "_old_forward"):
            tf_block.rotary_emb.forward = tf_block.rotary_emb._old_forward
            delattr(tf_block.rotary_emb, "_old_forward")
            

def register_cache_LLaDA(model: nn.Module, tf_block_module_key_name: str) -> None:

    target_module: Optional[nn.ModuleList] = None
    for name, module in model.named_modules():
        if name == tf_block_module_key_name:
            target_module = module
    for tf_block in target_module:
        setattr(tf_block, "_old_forward", tf_block.forward)
        tf_block.forward = types.MethodType(cache_hook_feature, tf_block)
        setattr(tf_block, "_old_attention", tf_block.attention)
        tf_block.attention = types.MethodType(_attention, tf_block)
        setattr(tf_block.rotary_emb, "_old_forward", tf_block.rotary_emb.forward)
        tf_block.rotary_emb.forward = types.MethodType(
            RoPe_forward, tf_block.rotary_emb
)


def _attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    q_index: torch.Tensor = None,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

    B, q_len, C = q.size()
    B, k_len, C = k.size()
    B, v_len, C = v.size()
    dtype = k.dtype
    if self.q_norm is not None and self.k_norm is not None:
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)
    q = q.view(B, q_len, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
    k = k.view(
        B, k_len, self.config.effective_n_kv_heads, C // self.config.n_heads
    ).transpose(1, 2)
    v = v.view(
        B, v_len, self.config.effective_n_kv_heads, C // self.config.n_heads
    ).transpose(1, 2)
    if layer_past is not None:
        past_key, past_value = layer_past
        k = torch.cat((past_key, k), dim=-2)
        v = torch.cat((past_value, v), dim=-2)
    present = (k, v) if use_cache else None
    query_len, key_len = q.shape[-2], k.shape[-2]
    if self.config.rope:
        q, k = self.rotary_emb(q, k, q_index=q_index)
    if attention_bias is not None:
        attention_bias = self._cast_attn_bias(
            attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
        )
    att = self._scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0 if not self.training else self.config.attention_dropout,
        is_causal=False,
    )
    att = att.transpose(1, 2).contiguous().view(B, q_len, C)
    return self.attn_out(att), present


def RoPe_forward(
    self, q: torch.Tensor, k: torch.Tensor, q_index: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.config.rope_full_precision:
        q_, k_ = q.float(), k.float()
    else:
        q_, k_ = q, k
    with torch.autocast(q.device.type, enabled=False):
        query_len, key_len = q_.shape[-2], k_.shape[-2]
        pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
        pos_sin = pos_sin.type_as(q_)
        pos_cos = pos_cos.type_as(q_)
        if q_index is not None:
            bs, _ = q_index.shape
            q_list = []
            for i in range(bs):
                q_i = self.apply_rotary_pos_emb(
                    pos_sin[:, :, q_index[i], :],
                    pos_cos[:, :, q_index[i], :],
                    q_[i].unsqueeze(0),
                )
                q_list.append(q_i)
            q_ = torch.cat(q_list, dim=0)
        else:
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
        k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
    return q_.type_as(q), k_.type_as(k)


def refresh_index(
    new_features: torch.Tensor,
    cached_features: torch.Tensor = None,
    transfer_ratio: float = 0.5,
    layer_id: int = 0,
) -> torch.Tensor:
    batch_size, gen_len, d_model = new_features.shape
    num_replace = int(gen_len * transfer_ratio)
    cos_sim = torch.nn.functional.cosine_similarity(
        new_features, cached_features, dim=-1
    )
    transfer_index = torch.topk(cos_sim, largest=False, k=num_replace).indices
    return transfer_index


def cache_hook_feature(
    self,
    x: torch.Tensor,
    attention_bias: Optional[torch.Tensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    feature_cache = dLLMCache()
    feature_cache.update_step(self.layer_id)
    prompt_length = feature_cache.prompt_length
    x_prompt = x[:, :prompt_length, :]
    x_gen = x[:, prompt_length:, :]
    refresh_gen = feature_cache.refresh_gen(layer_id=self.layer_id)
    refresh_prompt = feature_cache.refresh_prompt(layer_id=self.layer_id)
    transfer_ratio = feature_cache.transfer_ratio
    bs, seq_len, dim = x.shape
    transfer = transfer_ratio > 0 and transfer_ratio <= 1

    def attention(q, k, v, q_index: torch.Tensor = None):
        if self._activation_checkpoint_fn is not None:
            att, _ = self._activation_checkpoint_fn(
                self.attention,
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                q_index=q_index,
            )
        else:
            att, _ = self.attention(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                q_index=q_index,
            )
        return att

    def compute_mlp(input_x):

        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, input_x)
        else:
            x = self.ff_norm(input_x)
        x, x_up = self.ff_proj(x), self.up_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)
        else:
            x = self.act(x)
        x = x * x_up
        return self.ff_out(x)

    def project(x):
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)
        return q, k, v

    if refresh_gen and refresh_prompt:
        q, k, v = project(x)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="kv_cache",
            features={"k": k[:, :prompt_length, :], "v": v[:, :prompt_length, :]},
            cache_type="prompt",
        )
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="kv_cache",
            features={"k": k[:, prompt_length:, :], "v": v[:, prompt_length:, :]},
            cache_type="gen",
        )
        att = attention(q, k, v)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="attn",
            features=att[:, :prompt_length, :],
            cache_type="prompt",
        )
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="attn",
            features=att[:, prompt_length:, :],
            cache_type="gen",
        )

    elif refresh_gen and not refresh_prompt:
        q, k_gen, v_gen = project(x_gen)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="kv_cache",
            features={"k": k_gen, "v": v_gen},
            cache_type="gen",
        )
        kv_cache_prompt = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="kv_cache", cache_type="prompt"
        )
        k = torch.cat([kv_cache_prompt["k"], k_gen], dim=1)
        v = torch.cat([kv_cache_prompt["v"], v_gen], dim=1)
        att_gen = attention(q, k, v)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="attn",
            features=att_gen,
            cache_type="gen",
        )
        att_prompt_cache = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="attn", cache_type="prompt"
        )
        att = torch.cat([att_prompt_cache, att_gen], dim=1)

    elif not refresh_gen and refresh_prompt:
        q_prompt, k_prompt, v_prompt = project(x_prompt)
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="kv_cache",
            features={"k": k_prompt, "v": v_prompt},
            cache_type="prompt",
        )
        kv_cache_gen = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="kv_cache", cache_type="gen"
        )
        att_gen_cache = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="attn", cache_type="gen"
        )
        if transfer:
            x_gen_normed = self.attn_norm(x_gen)
            v_gen = self.v_proj(x_gen_normed)
            index = refresh_index(
                v_gen, kv_cache_gen["v"], transfer_ratio, self.layer_id
            )
            index_expanded = index.unsqueeze(-1).expand(-1, -1, dim)
            x_gen_selected = torch.gather(x_gen_normed, dim=1, index=index_expanded)
            q_gen_index = self.q_proj(x_gen_selected)
            k_gen_index = self.k_proj(x_gen_selected)
            kv_cache_gen["v"] = v_gen
            kv_cache_gen["k"].scatter_(dim=1, index=index_expanded, src=k_gen_index)
            feature_cache.set_cache(
                layer_id=self.layer_id,
                feature_name="kv_cache",
                features={"k": kv_cache_gen["k"], "v": kv_cache_gen["v"]},
                cache_type="gen",
            )
        k = torch.cat([k_prompt, kv_cache_gen["k"]], dim=1)
        v = torch.cat([v_prompt, kv_cache_gen["v"]], dim=1)
        if transfer:
            q_prompt_gen_index = torch.cat([q_prompt, q_gen_index], dim=1)
            prompt_index = (
                torch.arange(prompt_length)
                .unsqueeze(0)
                .expand(bs, -1)
                .to(q_prompt_gen_index.device)
            )
            gen_index = index + prompt_length
            att_prompt_gen_index = attention(
                q_prompt_gen_index,
                k,
                v,
                q_index=torch.cat([prompt_index, gen_index], dim=1),
            )
            att_prompt = att_prompt_gen_index[:, :prompt_length, :]
            att_gen_index = att_prompt_gen_index[:, prompt_length:, :]
            att_gen_cache.scatter_(dim=1, index=index_expanded, src=att_gen_index)
            feature_cache.set_cache(
                layer_id=self.layer_id,
                feature_name="attn",
                features=att_gen_cache,
                cache_type="gen",
            )
        else:
            att_prompt = attention(
                q_prompt,
                k,
                v,
                q_index=torch.arange(prompt_length).unsqueeze(0).expand(bs, -1),
            )
        feature_cache.set_cache(
            layer_id=self.layer_id,
            feature_name="attn",
            features=att_prompt,
            cache_type="prompt",
        )
        att = torch.cat([att_prompt, att_gen_cache], dim=1)
    else:
        att_gen_cache = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="attn", cache_type="gen"
        )
        if transfer:
            x_gen_normed = self.attn_norm(x_gen)
            v_gen = self.v_proj(x_gen_normed)
            kv_cache_gen = feature_cache.get_cache(
                layer_id=self.layer_id, feature_name="kv_cache", cache_type="gen"
            )
            kv_cache_prompt = feature_cache.get_cache(
                layer_id=self.layer_id, feature_name="kv_cache", cache_type="prompt"
            )
            index = refresh_index(
                v_gen, kv_cache_gen["v"], transfer_ratio, self.layer_id
            )
            index_expanded = index.unsqueeze(-1).expand(-1, -1, dim)
            x_gen_selected = torch.gather(x_gen_normed, dim=1, index=index_expanded)
            q_gen_index = self.q_proj(x_gen_selected)
            k_gen_index = self.k_proj(x_gen_selected)
            kv_cache_gen["v"] = v_gen
            kv_cache_gen["k"].scatter_(dim=1, index=index_expanded, src=k_gen_index)
            feature_cache.set_cache(
                layer_id=self.layer_id,
                feature_name="kv_cache",
                features={"k": kv_cache_gen["k"], "v": kv_cache_gen["v"]},
                cache_type="gen",
            )
            k = torch.cat([kv_cache_prompt["k"], kv_cache_gen["k"]], dim=1)
            v = torch.cat([kv_cache_prompt["v"], kv_cache_gen["v"]], dim=1)
            att_gen_index = attention(q_gen_index, k, v, q_index=index + prompt_length)
            att_gen_cache.scatter_(dim=1, index=index_expanded, src=att_gen_index)
            feature_cache.set_cache(
                layer_id=self.layer_id,
                feature_name="attn",
                features=att_gen_cache,
                cache_type="gen",
            )

        att_prompt_cache = feature_cache.get_cache(
            layer_id=self.layer_id, feature_name="attn", cache_type="prompt"
        )
        att = torch.cat([att_prompt_cache, att_gen_cache], dim=1)

    x = x + self.dropout(att)

    og_x = x
    x_prompt = x[:, :prompt_length, :]
    x_gen = x[:, prompt_length:, :]

    if refresh_gen and refresh_prompt:
        x = compute_mlp(x)
        feature_cache.set_cache(
            self.layer_id, "mlp", x[:, prompt_length:, :], cache_type="gen"
        )
        feature_cache.set_cache(
            self.layer_id, "mlp", x[:, :prompt_length, :], cache_type="prompt"
        )

    elif refresh_gen and not refresh_prompt:
        x_gen = compute_mlp(x_gen)
        feature_cache.set_cache(self.layer_id, "mlp", x_gen, cache_type="gen")
        x_prompt_cache = feature_cache.get_cache(
            self.layer_id, "mlp", cache_type="prompt"
        )
        x = torch.cat([x_prompt_cache, x_gen], dim=1)

    elif refresh_prompt and not refresh_gen:
        x_gen_cache = feature_cache.get_cache(self.layer_id, "mlp", cache_type="gen")
        if transfer:
            x_gen_selected = torch.gather(x_gen, dim=1, index=index_expanded)
            x_prompt_gen_index = torch.cat([x_prompt, x_gen_selected], dim=1)
            x_prompt_gen_index = compute_mlp(x_prompt_gen_index)
            x_prompt = x_prompt_gen_index[:, :prompt_length, :]
            x_gen_index = x_prompt_gen_index[:, prompt_length:, :]
            x_gen_cache.scatter_(dim=1, index=index_expanded, src=x_gen_index)
            feature_cache.set_cache(self.layer_id, "mlp", x_gen_cache, cache_type="gen")
        else:
            x_prompt = compute_mlp(x_prompt)
        feature_cache.set_cache(self.layer_id, "mlp", x_prompt, cache_type="prompt")
        x = torch.cat([x_prompt, x_gen_cache], dim=1)

    else:
        x_gen_cache = feature_cache.get_cache(self.layer_id, "mlp", cache_type="gen")
        if transfer:
            x_gen_selected = torch.gather(x_gen, dim=1, index=index_expanded)
            x_gen_index = compute_mlp(x_gen_selected)
            x_gen_cache.scatter_(dim=1, index=index_expanded, src=x_gen_index)
            feature_cache.set_cache(self.layer_id, "mlp", x_gen_cache, cache_type="gen")
        x_prompt_cache = feature_cache.get_cache(
            self.layer_id, "mlp", cache_type="prompt"
        )
        x = torch.cat([x_prompt_cache, x_gen_cache], dim=1)

    x = self.dropout(x)
    x = og_x + x

    return x, None
