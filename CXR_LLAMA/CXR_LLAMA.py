import torch, transformers
import json, os
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from open_clip.model import _build_vision_tower
from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

USE_MODIFIED_INPUT_LABELS = True
USE_BFLOAT16_IMAGE_ENCODER = True

class CXR_LLAMA_Loader():
    def __init__(self, model_path, temperature =0, top_p=1):
        if not torch.cuda.is_available():
            raise Exception("You must run model on GPU.")

        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(model_path, "config.json"), 'r', encoding="utf-8") as f:
            json_object = json.load(f)
            json_object['llama_model_path'] = model_path
        config = CXR_LLAMA_Config(**json_object)
        self.model = CXR_LLAMA_Model.from_pretrained(model_path, config=config)
        self.model.to(self.device)
        self.temperature = temperature
        self.top_p = top_p
    def apply_chat_template(self, chat):
        return self.model.tokenizer.apply_chat_template(chat, tokenize=False)
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

    def generate(self, chat, pil_image):
        if self.model is None:
            raise Exception("CXR_LLAMA Model is not loaded")

        prompt = self.apply_chat_template(chat)
        images = self.model.vision_tower.image_processor(pil_image, return_tensors='pt')['pixel_values']
        input_ids = self.tokenizer_image_token(prompt, self.model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_args = {"images": images}
        do_sample = True if self.temperature > 0.001 else False
        num_image_tokens = 1
        max_context_length = 4096
        max_new_tokens = min(4096, max_context_length - input_ids.shape[-1] - num_image_tokens)

        result = self.model.generate(inputs=input_ids,
                            do_sample=do_sample,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            max_new_tokens=max_new_tokens,
                            use_cache=True,
                            **image_args)

        with open(r'D:\ResearchFire\CXR-LLAMA\result.pickle', 'wb') as f:
            import pickle
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

        result_string = self.model.tokenizer.decode(result[0])
        return result_string

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

class CXR_LLAMA_Config(PretrainedConfig):
    model_type = "CXR_LLAMA"

    def __init__(
            self,
            llama_model_path=None,  #
            llama_model_dtype='bf16',
            clip_vision_tower_path=None,
            clip_vision_tower_dtype='bf16',
            clip_vision_cfg=None,
            clip_embed_dim=1024,
            image_preprocess_cfg=None,
            mm_projector_dim=1024,
            mm_projector_path=None,
            mm_projector_dtype='bf16',
            llama=None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        if clip_vision_cfg is None:
            clip_vision_cfg = {
                "image_size": 1024,
                "layers": [3, 4, 6, 3],
                "width": 64,
                "patch_size": None
            }

        if image_preprocess_cfg is None:
            image_preprocess_cfg = transformers.CLIPImageProcessor(
                do_resize=True,
                size={'shortest_edge': clip_vision_cfg['image_size']},
                resample=True,
                do_center_crop=True,
                crop_size=clip_vision_cfg['image_size'],
                do_rescale=True,
                rescale_factor=1 / 255,
                do_normalize=True,
                image_mean=0.5518136078431373,
                image_std=0.3821719215686275,
                do_convert_rgb=True
            )

        self.clip_vision_cfg = clip_vision_cfg
        self.clip_vision_tower_path = clip_vision_tower_path
        self.clip_embed_dim = clip_embed_dim
        self.clip_quick_gelu = False
        self.clip_vision_tower_dtype = clip_vision_tower_dtype
        self.image_preprocess_cfg = image_preprocess_cfg

        self.llama_model_path = llama_model_path

        if llama is not None:
            self.llama = transformers.LlamaConfig(**llama)

        else:
            if llama_model_path is not None:
                self.llama = transformers.LlamaConfig.from_pretrained(llama_model_path)

        self.llama_model_dtype = llama_model_dtype

        self.mm_projector_dim = mm_projector_dim
        self.mm_projector_path = mm_projector_path
        self.mm_projector_dtype = mm_projector_dtype



class CXR_LLAMA_Model(PreTrainedModel):
    config_class = CXR_LLAMA_Config

    def __init__(self, config):
        super().__init__(config)

        if type(config.clip_vision_cfg) == str:
            config.clip_vision_cfg = json.loads(config.clip_vision_cfg)

        if type(config.image_preprocess_cfg) == str:
            config.image_preprocess_cfg = json.loads(config.image_preprocess_cfg)

        self.vision_tower = _build_vision_tower(config.clip_embed_dim, config.clip_vision_cfg, config.clip_quick_gelu,
                                                config.clip_vision_tower_dtype)
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
            do_convert_rgb=True
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

        if config.clip_vision_tower_path is not None:
            pretrained_vision_tower = torch.load(config.clip_vision_tower_path)
            if 'state_dict' in pretrained_vision_tower.keys():
                pretrained_vision_tower = pretrained_vision_tower['state_dict']
            new_state_dict = {}
            for key in pretrained_vision_tower.keys():
                newkey = key.replace("model.visual.", "")
                newkey = newkey.replace("visual.trunk", "trunk")
                new_state_dict[newkey] = pretrained_vision_tower[key]
            intersect_state_keys = set(self.vision_tower.state_dict().keys()).intersection(new_state_dict.keys())
            if len(intersect_state_keys) == 0:
                raise Exception("Cannot apply pretrained vision tower weight")
            self.vision_tower.load_state_dict(new_state_dict, strict=False)

        self.mm_projector = torch.nn.Linear(config.mm_projector_dim, config.llama.hidden_size)
        self.lm_head = torch.nn.Linear(config.llama.hidden_size, config.llama.vocab_size, bias=False)

        # loading lm_head
        json_path = os.path.join(config.llama_model_path, "pytorch_model.bin.index.json")
        new_state_dict = {}
        with open(json_path, 'r', encoding="utf-8") as f:
            json_object = json.load(f)
            for key in json_object['weight_map'].keys():
                if "lm_head" in key:
                    lm_head_weight_path = os.path.join(config.llama_model_path, json_object['weight_map'][key])

                    pretrained_lm_weights = torch.load(lm_head_weight_path, map_location='cpu')
                    for key in pretrained_lm_weights.keys():
                        if "lm_head" in key:
                            new_state_dict[key.replace("lm_head.", "")] = pretrained_lm_weights[key]

        if len(new_state_dict.keys()) == 0:
            raise Exception("No LM HEAD weight found")

        self.lm_head.load_state_dict(new_state_dict, strict=True)
        del new_state_dict

        self.llama = transformers.LlamaModel(config.llama)
        '''
        if config.llama_model_path is not None:
            self.llama = transformers.LlamaModel.from_pretrained(config.llama_model_path, config=config.llama)
        else:
            raise Exception("Cannot load LLAMA base model")
        '''
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(config.llama_model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token

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
        images = images.to(torch.bfloat16).to(self.device)
        image_features = self.vision_tower(images)
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

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal_use_final_vector(
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
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.llama.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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