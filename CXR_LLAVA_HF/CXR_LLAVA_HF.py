from transformers import PretrainedConfig, PreTrainedModel
import torch, transformers
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from .VisualTransformer import  VisionTransformer, LayerNorm
from functools import partial
from transformers import TextIteratorStreamer
from transformers import StoppingCriteria, GenerationConfig
from threading import Thread
from dataclasses import dataclass
import numpy as np
from PIL import Image
# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'AttrDict' object has no attribute '{key}'")


class CXRLLAVAConfig(PretrainedConfig):
    model_type = "CXR-LLAVA"

    def __init__(self,  **kwargs,):

        if 'llama' in kwargs:
            self.llama = AttrDict(kwargs['llama'])
            del kwargs['llama']

        self.__dict__.update(kwargs)
        super().__init__(**kwargs)


class CXRLLAVAModel(PreTrainedModel):
    config_class = CXRLLAVAConfig

    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(config._name_or_path, add_special_tokens=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.sep_token = self.tokenizer.unk_token
        self.tokenizer.cls_token = self.tokenizer.unk_token
        self.tokenizer.mask_token = self.tokenizer.unk_token

        vision_cfg = CLIPVisionCfg(**config.clip_vision_cfg)

        self.generation_config =  GenerationConfig.from_pretrained(config._name_or_path)

        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNorm
        act_layer = torch.nn.GELU
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        self.vision_tower = VisionTransformer(
            in_channels=1,
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=config.clip_embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.vision_tower.image_processor = transformers.CLIPImageProcessor(
            do_resize=True,
            size={'shortest_edge': config.clip_vision_cfg['image_size']},
            resample=True,
            do_center_crop=True,
            crop_size=config.clip_vision_cfg['image_size'],
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=config.image_preprocess_cfg['mean'],
            image_std=config.image_preprocess_cfg['std'],
            do_convert_rgb=False
        )

        def convert_dtype(dtype):
            if dtype == 'fp32':
                dtype = torch.float32
            elif dtype == 'fp16':
                dtype = torch.float16
            elif dtype == 'bf16':
                dtype = torch.bfloat16
            else:
                raise Exception("Unsupported dtype")
            return dtype

        self.clip_cast_dtype = convert_dtype(config.clip_vision_tower_dtype)
        self.mm_projector = torch.nn.Linear(config.mm_projector_dim, config.llama['hidden_size'])
        self.lm_head = torch.nn.Linear(config.llama.hidden_size, config.llama.vocab_size, bias=False)
        self.llama = transformers.LlamaModel(transformers.LlamaConfig(**config.llama))

        self.llama = self.llama.to(torch.bfloat16)
        self.lm_head = self.lm_head.to(torch.bfloat16)
        self.vision_tower = self.vision_tower.to(torch.bfloat16)
        self.mm_projector = self.mm_projector.to(torch.bfloat16)

    def get_input_embeddings(self):
        return self.llama.get_input_embeddings()

    def get_vision_tower(self):
        return self.vision_tower

    def gradient_checkpointing_enable(self):
        return self.llama.gradient_checkpointing_enable()

    def encode_images(self, images):
        images = images.to(torch.bfloat16)

        def _expand_token(token, batch_size: int):
            return token.view(1, 1, -1).expand(batch_size, -1, -1)

        # open_clip ViT
        # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
        x = images
        x = self.vision_tower.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.vision_tower.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.vision_tower.positional_embedding.to(x.dtype)

        x = self.vision_tower.patch_dropout(x)
        x = self.vision_tower.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.vision_tower.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.vision_tower.attn_pool is not None:
            if self.vision_tower.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.vision_tower.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.vision_tower.attn_pool(x)
                if self.vision_tower.attn_pool_type == 'parallel':
                    pooled = self.vision_tower.attn_pool_contrastive(x)
                else:
                    assert self.vision_tower.attn_pool_type == 'cascade'
                    pooled = self.vision_tower.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.vision_tower.attn_pool(x)
                x = self.vision_tower.ln_post(x)
                pooled, tokens = self.vision_tower._global_pool(x)
        elif self.vision_tower.final_ln_after_pool:
            pooled, tokens = self.vision_tower._global_pool(x)
            pooled = self.vision_tower.ln_post(pooled)
        else:
            x = self.vision_tower.ln_post(x)
            pooled, tokens = self.vision_tower._global_pool(x)

        if self.vision_tower.proj is not None:
            pooled = pooled @ self.vision_tower.proj

        image_features = tokens
        image_features = image_features.to(torch.bfloat16)
        image_features = self.mm_projector(image_features)

        image_features = image_features.to(torch.bfloat16)
        return image_features

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,  # (1,4317)
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images)

        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # original multimodal code
    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.llama.embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (0. * self.mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_new_input_embeds.append(self.llama.embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                    cur_new_input_embeds.append(
                        self.llama.embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.llama.embed_tokens(cur_input_ids[image_token_start + 1:image_token_start + 2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start + 1])
                        cur_labels = cur_labels[image_token_start + 2:]
                else:
                    cur_new_input_embeds.append(self.llama.embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start + 1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_input_ids = cur_input_ids[image_token_start + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_new_input_embeds.append(self.llama.embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.llama.embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    # sw-modified code

    def prepare_inputs_labels_for_multimodal_use_final_vector(
            self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.llama.embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (0. * self.mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_new_input_embeds.append(self.llama.embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                    cur_new_input_embeds.append(
                        self.llama.embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.llama.embed_tokens(cur_input_ids[image_token_start + 1:image_token_start + 2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start + 1])
                        cur_labels = cur_labels[image_token_start + 2:]
                else:
                    cur_new_input_embeds.append(
                        self.llama.embed_tokens(cur_input_ids[:image_token_start].to(self.device)))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start + 1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_input_ids = cur_input_ids[image_token_start + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_new_input_embeds.append(self.llama.embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.llama.embed_tokens(cur_input_ids.to(self.device)))
                if labels is not None:
                    # seowoo-edit
                    cur_labels = labels[batch_idx]
                    cur_new_labels.append(cur_labels)
            # [5120] -> [1, 5120]
            cur_new_input_embeds[1] = torch.unsqueeze(cur_new_input_embeds[1], dim=0)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            # print("if 204")
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

            return None, attention_mask, past_key_values, new_input_embeds, labels

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def apply_chat_template(self, chat):
        return self.tokenizer.apply_chat_template(chat, tokenize=False)

    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def write_radiologic_report(self, image, temperature=0.2, top_p=0.8):
        chat = [
            {"role": "system",
             "content": "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides."},
            {"role": "user",
             "content": "<image>\nWrite a radiologic report on the given chest radiograph, including information about atelectasis, cardiomegaly, consolidation, pulmonary edema, pleural effusion, and pneumothorax.\n"}
        ]
        response = self.generate_cxr_repsonse(chat=chat,image=image, temperature=temperature, top_p=top_p)
        return response

    def write_differential_diagnosis(self, image, temperature=0.2, top_p=0.8):
        chat = [
            {"role": "system",
             "content": "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides."},
            {"role": "user",
             "content": "<image>\nWhat are the possible differential diagnoses for this patient?\n"}
        ]
        response = self.generate_cxr_repsonse(chat=chat, image=image, temperature=temperature, top_p=top_p)
        return response
    def ask_question(self, question, image, temperature=0.2, top_p=0.8):
        chat = [
            {"role": "system",
             "content": "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides."},
            {"role": "user",
             "content": "<image>\n"+question}
        ]
        response = self.generate_cxr_repsonse(chat=chat, image=image, temperature=temperature, top_p=top_p)
        return response

    def generate_cxr_repsonse(self, chat, image, temperature=0.2, top_p=0.8):
        with torch.no_grad():
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

            if np.array(image).max()>255:
                raise Exception("WARNING. 16-bit image is not supported.")

            image = image.convert('L') # convert to grayscale
            image = np.array(image)

            if len(image.shape) == 2:
                image = np.expand_dims(image,axis=-1) # (width, height) --> (width, height, 1)

            prompt = self.apply_chat_template(chat)
            images = self.vision_tower.image_processor(image, return_tensors='pt')['pixel_values']
            images = images.to(self.device)
            input_ids = self.tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stopping_criteria = KeywordsStoppingCriteria(["</s>"], self.tokenizer, input_ids)

            image_args = {"images": images}
            do_sample = True if temperature > 0.001 else False
            num_image_tokens = 1
            max_context_length = getattr(self.config, 'max_position_embeddings', 2048)

            max_new_tokens = min(512, max_context_length - input_ids.shape[-1] - num_image_tokens)
            thread = Thread(target=self.generate, kwargs=dict(
                inputs=input_ids,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                stopping_criteria=[stopping_criteria],
                use_cache=True,
                generation_config=self.generation_config,
                **image_args
            ))
            thread.start()
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text

        return generated_text

    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    timm_model_name: Optional[str] = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth
