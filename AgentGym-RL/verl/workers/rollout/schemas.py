from dataclasses import dataclass
from typing import List, Literal, Optional
from transformers import PreTrainedTokenizer
import torch


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    def to_dict(self):
        return {'role': self.role, 'content': self.content}
    def __repr__(self):
        return str(self.to_dict())
    def __str__(self):
        return self.__repr_

class RolloutHandler:
    def __init__(
        self,
        messages: List[Message],
        task_name: str,
        item_id: int,
        score: float,
        done: bool,
        input_ids: List[int],
        prompt_ids: List[int],
        response_ids: List[int],
        attention_mask: List[int],
        prompt_attention_mask: List[int],
        response_attention_mask: List[int],
        position_ids: List[int],
        prompt_position_ids: List[int],
        response_position_ids: List[int],
        loss_mask: List[int],
        prompt_loss_mask: List[int],
        response_loss_mask: List[int],
        max_response_len: int = 8192,
        max_model_len: int = 32768
    ):
        self.messages = messages
        self.task_name = task_name
        self.item_id = item_id
        self.score = score
        self.done = done
        self.input_ids = input_ids
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.attention_mask = attention_mask
        self.prompt_attention_mask = prompt_attention_mask
        self.response_attention_mask = response_attention_mask
        self.position_ids = position_ids
        self.prompt_position_ids = prompt_position_ids
        self.response_position_ids = response_position_ids
        self.loss_mask = loss_mask
        self.prompt_loss_mask = prompt_loss_mask
        self.response_loss_mask = response_loss_mask
        self.max_response_len = max_response_len
        self.max_model_len = max_model_len
        self.format_config: dict = {
            "qwen": {
                "assistat_prefix_msg": "\n<|im_start|>assistant\n",
                "assistat_suffix_msg": "<|im_end|>",
                "user_prefix_msg": "\n<|im_start|>user\n",
                "user_suffix_msg": "<|im_end|>",
            }
        }

    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer, memory_examples: Optional[List[dict]] = None) -> List[int]:
        """
        Generate prompt with optional memory examples injected before current conversation.

        Args:
            tokenizer: Tokenizer for encoding
            memory_examples: Optional list of few-shot examples from memory bank
                            Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

        Returns:
            Tokenized prompt with memory examples prepended
        """
        conversations = [
            msg.to_dict() for msg in self.messages
        ]

        # Inject memory examples before current conversation if provided
        if memory_examples and len(memory_examples) > 0:
            # Find system message if it exists
            system_msg = None
            current_conv = conversations
            if conversations and conversations[0].get("role") == "system":
                system_msg = conversations[0]
                current_conv = conversations[1:]

            # Build conversation with memory examples
            if system_msg:
                conversations = [system_msg] + memory_examples + current_conv
            else:
                conversations = memory_examples + current_conv

        return tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True)


    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        format: Literal["qwen"] = "qwen",
    ) -> None:
        msg = Message(role='assistant', content=content)
        self.messages.append(msg)
        assert format in self.format_config.keys(), f"format {format} not supported"
        prefix_msg = self.format_config[format]["assistat_prefix_msg"]
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_msg = self.format_config[format]["assistat_suffix_msg"]
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        response = tokenizer.encode(content, add_special_tokens=False)
        if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids:
            append_token_ids = response
            _loss_mask = [1] * len(response)
        elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
            append_token_ids = prefix_token_ids + response
            _loss_mask = [0] * len(prefix_token_ids) + [1] * len(response)
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(
                f"""Unsupported end of message format:
                {tokenizer.decode(self.input_ids[-max_len:])}, {tokenizer.decode(self.input_ids)=}"""
            )
        append_token_ids += suffix_token_ids
        _loss_mask += [1] * len(suffix_token_ids)
        self.input_ids += append_token_ids
        _attention_mask = [1] * len(append_token_ids)
        self.attention_mask += _attention_mask
        _delta_position_ids = [pos_id for pos_id in range(1, len(append_token_ids) + 1)]
        last_position_ids = self.position_ids[-1]
        _position_ids = [pos_id + last_position_ids for pos_id in _delta_position_ids]
        self.loss_mask += _loss_mask
        self.position_ids += _position_ids
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Rollout Handler has different length of {len(self.input_ids)=},
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def add_user_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        format: Literal["qwen"] = "qwen",
    ) -> None:
        msg = Message(role='user', content=content)
        self.messages.append(msg)
        assert format in self.format_config.keys(), f"format {format} not supported"
        prefix_msg = self.format_config[format]["user_prefix_msg"]
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_msg = self.format_config[format]["user_suffix_msg"]
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        content_token_ids = tokenizer.encode(content, add_special_tokens=False)

        if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids:
            append_token_ids = content_token_ids
            _loss_mask = [0] * len(content_token_ids)
        elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
            append_token_ids = prefix_token_ids + content_token_ids
            _loss_mask = [0] * len(prefix_token_ids) + [0] * len(content_token_ids)
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(
                f"""Unsupported end of message format:
                {tokenizer.decode(self.input_ids[-max_len:])}, {tokenizer.decode(self.input_ids)=}"""
            )

        append_token_ids += suffix_token_ids
        _loss_mask += [0] * len(suffix_token_ids)
        self.input_ids += append_token_ids
        _attention_mask = [1] * len(append_token_ids)
        self.attention_mask += _attention_mask
        _delta_position_ids = [pos_id for pos_id in range(1, len(append_token_ids) + 1)]
        last_position_ids = self.position_ids[-1]
        _position_ids = [pos_id + last_position_ids for pos_id in _delta_position_ids]
        self.loss_mask += _loss_mask
        self.position_ids += _position_ids
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Rollout Handler has different length of {len(self.input_ids)=},
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def truncate_output_ids(self) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
