"""
Eric's Semantic Search - Qwen3-VL Embedding

Description: Implementation of the Qwen3-VL model for embedding extraction with batch
             processing support. Based on official Qwen code, adapted for ComfyUI
             integration with performance optimizations.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

Third-party code:
- Based on Qwen3-VL by Alibaba Qwen Team (Qwen License)
  https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B

Dependencies:
- qwen-vl-utils>=0.0.14 (Apache 2.0): Qwen vision-language utilities
- Transformers (Apache 2.0): Hugging Face
- PyTorch (BSD License): Meta Platforms

See LICENSE.md for complete license information.
"""

import torch
import torch.nn.functional as F
import unicodedata
import numpy as np
import logging
import time

from PIL import Image
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel, Qwen3VLModel, Qwen3VLConfig
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants - these are defaults, can be overridden by caller
MAX_LENGTH = 8192
IMAGE_FACTOR = 32
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR  # 4096
MAX_PIXELS = 1024 * 1024  # 1MP default for good accuracy


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Qwen3-VL model modified for embedding extraction (no LM head)"""
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_image_features(self, pixel_values: torch.FloatTensor,
                           image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Cache] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                pixel_values: Optional[torch.Tensor] = None,
                pixel_values_videos: Optional[torch.FloatTensor] = None,
                image_grid_thw: Optional[torch.LongTensor] = None,
                video_grid_thw: Optional[torch.LongTensor] = None,
                cache_position: Optional[torch.LongTensor] = None,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLForEmbeddingOutput]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


def _check_qwen_vl_utils_version():
    """Check if qwen-vl-utils supports Qwen3-VL (>=0.0.14)"""
    try:
        import importlib.metadata
        version = importlib.metadata.version('qwen-vl-utils')
        print(f"[SemanticSearch] qwen-vl-utils version: {version}")
        parts = version.split('.')[:3]
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        is_supported = (major, minor, patch) >= (0, 0, 14)
        print(f"[SemanticSearch] Qwen3-VL batch support: {is_supported}")
        return is_supported
    except Exception as e:
        print(f"[SemanticSearch] Could not determine qwen-vl-utils version: {e}")
        return False


class Qwen3VLEmbedder():
    """
    Embedding extractor for Qwen3-VL with full batch processing support.
    """
    
    def __init__(
        self, 
        model_name_or_path: str, 
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        default_instruction: str = "Represent the user's input.",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",  # sdpa or eager
        **kwargs
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.default_instruction = default_instruction
        self.torch_dtype = torch_dtype
        self._logged_timing = False
        
        print(f"[SemanticSearch] Device: {self.device}")
        if torch.cuda.is_available():
            print(f"[SemanticSearch] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[SemanticSearch] CUDA version: {torch.version.cuda}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[SemanticSearch] GPU Memory: {total_mem:.1f} GB")
            cc = torch.cuda.get_device_capability(0)
            print(f"[SemanticSearch] Compute Capability: {cc[0]}.{cc[1]}")
        else:
            print("[SemanticSearch] WARNING: CUDA not available!")
        
        # Check qwen-vl-utils version
        self._has_qwen3_support = _check_qwen_vl_utils_version()
        
        # Handle attention implementation
        actual_attn_impl = attn_implementation
        
        # Sage attention requires special integration - not simple monkey-patch
        # For now, fall back to sdpa if sage is requested
        if attn_implementation == "sage":
            print("[SemanticSearch] SageAttention requested but requires custom integration.")
            print("[SemanticSearch] Falling back to sdpa. For sage support, see SageAttention docs.")
            actual_attn_impl = "sdpa"
        elif attn_implementation == "flash_attention_2":
            print("[SemanticSearch] flash_attention_2 not supported on Blackwell, using sdpa")
            actual_attn_impl = "sdpa"
        
        print(f"[SemanticSearch] Loading model with attn_implementation={actual_attn_impl}, dtype={torch_dtype}")
        print(f"[SemanticSearch] Image resolution: min={min_pixels}, max={max_pixels} pixels (~{int(max_pixels**0.5)}x{int(max_pixels**0.5)})")
        
        # Load model
        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=actual_attn_impl,
            **kwargs
        ).to(self.device)
        
        # Verify model is on GPU
        param_device = next(self.model.parameters()).device
        param_dtype = next(self.model.parameters()).dtype
        print(f"[SemanticSearch] Model loaded on: {param_device}, dtype: {param_dtype}")
        
        # Load processor with the ACTUAL max_pixels we want
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, 
            padding_side='right',
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        
        # Get patch size from processor for process_vision_info
        self.patch_size = getattr(self.processor.image_processor, 'patch_size', 16)
        
        self.model.eval()
        print(f"[SemanticSearch] Qwen3VLEmbedder initialized")

    def _load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """Load an image from various sources"""
        if isinstance(image_source, Image.Image):
            return image_source.convert('RGB')
        
        if isinstance(image_source, str):
            if image_source.startswith('file://'):
                image_source = image_source[7:]
            return Image.open(image_source).convert('RGB')
        
        raise TypeError(f"Unsupported image type: {type(image_source)}")

    def _format_instruction(self, instruction: Optional[str]) -> str:
        """Ensure instruction ends with punctuation"""
        if not instruction:
            return self.default_instruction
        
        instruction = instruction.strip()
        if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
            instruction = instruction + '.'
        return instruction

    def _build_conversation(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Image.Image]] = None,
        video: Optional[Union[str, List[Image.Image]]] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict]:
        """Build a single conversation for the chat template.
        
        Args:
            text: Optional text input
            image: Optional single image (path, URL, or PIL Image)
            video: Optional video - either a file path or list of PIL Images (frames)
            instruction: Optional instruction override
        """
        instruction = self._format_instruction(instruction)
        content = []
        
        # Video takes precedence over image if both provided
        if video is not None:
            if isinstance(video, str):
                # Video file path
                video_ref = video if video.startswith(('http', 'file://')) else f'file://{video}'
                content.append({
                    "type": "video", 
                    "video": video_ref,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                })
            elif isinstance(video, list) and len(video) > 0:
                # List of PIL Image frames - pass as images for multi-frame embedding
                # Qwen3-VL treats multiple images as a sequence similar to video
                for frame in video:
                    content.append({
                        "type": "image", 
                        "image": frame,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    })
        elif image is not None:
            if isinstance(image, str):
                img_ref = image if image.startswith(('http', 'file://')) else f'file://{image}'
            else:
                img_ref = image
            content.append({
                "type": "image", 
                "image": img_ref,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            })
        
        if text:
            content.append({"type": "text", "text": text})
        elif not image and not video:
            content.append({"type": "text", "text": "NULL"})
        
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content}
        ]

    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract the last non-padding token's hidden state for each sequence"""
        flipped = attention_mask.flip(dims=[1])
        last_positions = flipped.argmax(dim=1)
        col_indices = attention_mask.shape[1] - last_positions - 1
        row_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row_indices, col_indices]

    @torch.no_grad()
    def _process_batch_with_qwen_vl_utils(
        self, 
        conversations: List[List[Dict]]
    ) -> Dict[str, torch.Tensor]:
        """Process a batch using qwen-vl-utils (requires >=0.0.14)"""
        from qwen_vl_utils import process_vision_info
        
        texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]
        
        images, video_inputs, video_kwargs = process_vision_info(
            conversations,
            image_patch_size=self.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            videos, video_metadata = None, None
        
        inputs = self.processor(
            text=texts,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            # Note: do_resize=True (default) is required for max_pixels to work!
            # Previously had do_resize=False which bypassed resizing entirely.
            **video_kwargs
        )
        
        return inputs

    @torch.no_grad()
    def _process_single(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Image.Image]] = None,
        video: Optional[Union[str, List[Image.Image]]] = None,
        instruction: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Process a single input (image, video, or text)"""
        instruction = self._format_instruction(instruction)
        
        content = []
        pil_images = None
        
        # Video takes precedence (either path or list of frames)
        if video is not None:
            if isinstance(video, str):
                # Video file path - use qwen-vl-utils if available
                content.append({"type": "video"})
                # For single processing with video path, we need to load frames
                # This fallback path should use the batch processor
                raise NotImplementedError("Single video file processing requires batch mode. Use batch processing.")
            elif isinstance(video, list) and len(video) > 0:
                # List of PIL frames - treat as multiple images
                for _ in video:
                    content.append({"type": "image"})
                pil_images = [frame.convert('RGB') if isinstance(frame, Image.Image) else self._load_image(frame) for frame in video]
        elif image is not None:
            content.append({"type": "image"})
            pil_images = [self._load_image(image)]
        
        if text:
            content.append({"type": "text", "text": text})
        elif not image and not video:
            content.append({"type": "text", "text": "NULL"})
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content}
        ]
        
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        inputs = self.processor(
            text=[prompt],
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )
        
        return inputs

    @torch.no_grad()
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        """
        Process a batch of inputs and return embeddings.
        """
        batch_size = len(inputs)
        
        if batch_size == 0:
            raise ValueError("Empty input batch")
        
        # Try batch processing with qwen-vl-utils >= 0.0.14
        if self._has_qwen3_support and batch_size > 1:
            try:
                conversations = [
                    self._build_conversation(
                        text=inp.get('text'),
                        image=inp.get('image'),
                        video=inp.get('video'),
                        instruction=inp.get('instruction'),
                    )
                    for inp in inputs
                ]
                
                t0 = time.time()
                processed = self._process_batch_with_qwen_vl_utils(conversations)
                processed = {k: v.to(self.device) for k, v in processed.items()}
                t1 = time.time()
                
                # Log tensor info
                if not self._logged_timing:
                    for k, v in processed.items():
                        if isinstance(v, torch.Tensor):
                            print(f"[SemanticSearch] {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                
                # Forward pass
                outputs = self.model(**processed)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                embeddings = self._pooling_last(outputs.last_hidden_state, processed['attention_mask'])
                t2 = time.time()
                
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=-1)
                
                if not self._logged_timing:
                    print(f"[SemanticSearch] Batch({batch_size}) timing: preprocess={t1-t0:.2f}s, forward={t2-t1:.2f}s")
                    self._logged_timing = True
                
                return embeddings
                
            except Exception as e:
                print(f"[SemanticSearch] Batch processing failed: {e}. Falling back to single-item mode.")
                import traceback
                traceback.print_exc()
        
        # Fallback: process one at a time
        all_embeddings = []
        for i, inp in enumerate(inputs):
            try:
                t0 = time.time()
                processed = self._process_single(
                    text=inp.get('text'),
                    image=inp.get('image'),
                    video=inp.get('video'),
                    instruction=inp.get('instruction'),
                )
                processed = {k: v.to(self.device) for k, v in processed.items()}
                t1 = time.time()
                
                # Log tensor info for first image
                if i == 0 and not self._logged_timing:
                    for k, v in processed.items():
                        if isinstance(v, torch.Tensor):
                            print(f"[SemanticSearch] {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                
                outputs = self.model(**processed)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                emb = self._pooling_last(outputs.last_hidden_state, processed['attention_mask'])
                all_embeddings.append(emb)
                t2 = time.time()
                
                if i == 0 and not self._logged_timing:
                    print(f"[SemanticSearch] Single image timing: preprocess={t1-t0:.2f}s, forward={t2-t1:.2f}s")
                    self._logged_timing = True
                
            except Exception as e:
                print(f"[SemanticSearch] Failed to encode input {i}: {e}")
                raise
        
        embeddings = torch.cat(all_embeddings, dim=0)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
