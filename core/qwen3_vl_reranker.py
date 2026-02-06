"""
Eric's Semantic Search - Qwen3-VL Reranker

Description: Implementation of the Qwen3-VL model for reranking search results.
             Provides precise relevance scoring using cross-attention between
             queries and candidate images for two-stage retrieval.
             
Author: Eric Hiss (GitHub: EricRollei)
Contact: eric@historic.camera, eric@rollei.us
License: Dual License (Non-Commercial: CC BY-NC 4.0, Commercial: Contact author)
Copyright (c) 2026 Eric Hiss. All rights reserved.

Third-party code:
- Based on Qwen3-VL by Alibaba Qwen Team (Qwen License)
  https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B

Dependencies:
- qwen-vl-utils (Apache 2.0): Qwen vision-language utilities
- Transformers (Apache 2.0): Hugging Face
- PyTorch (BSD License): Meta Platforms
- SciPy (BSD License): Scientific computing

See LICENSE.md for complete license information.
"""

import torch
import numpy as np
import logging
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from PIL import Image
from scipy import special
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


@dataclass
class ProfileStats:
    """Accumulates timing statistics for performance profiling"""
    timings: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    def record(self, stage: str, duration: float):
        self.timings[stage].append(duration)
    
    def summary(self) -> str:
        lines = ["\n" + "="*60, "RERANKER PERFORMANCE PROFILE", "="*60]
        total_time = 0
        for stage, times in self.timings.items():
            avg = np.mean(times)
            total = np.sum(times)
            total_time += total
            lines.append(f"  {stage:30s}: {avg*1000:8.2f}ms avg × {len(times):4d} calls = {total:8.2f}s total")
        lines.append("-"*60)
        lines.append(f"  {'TOTAL':30s}: {total_time:8.2f}s")
        if self.timings.get('total_per_doc'):
            n_docs = len(self.timings['total_per_doc'])
            lines.append(f"  {'THROUGHPUT':30s}: {n_docs/total_time:.2f} docs/sec ({total_time/n_docs:.2f}s per doc)")
        lines.append("="*60 + "\n")
        return "\n".join(lines)
    
    def reset(self):
        self.timings = defaultdict(list)

logger = logging.getLogger(__name__)

MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR  # 4 tokens
MAX_PIXELS = 512 * 512  # Reduced for speed - reranker uses thumbnails anyway
MAX_RATIO = 200

FRAME_FACTOR = 2
FPS = 1
MIN_FRAMES = 2
MAX_FRAMES = 64
MIN_TOTAL_PIXELS = 1 * FRAME_FACTOR * MIN_PIXELS  # 1 frames
MAX_TOTAL_PIXELS = 4 * FRAME_FACTOR * MAX_PIXELS  # 4 frames


def sample_frames(frames, num_segments, max_segments):
    duration = len(frames)
    frame_id_array = np.linspace(0, duration - 1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    last_frame_id = frame_id_list[-1]

    sampled_frames = []
    for frame_idx in frame_id_list:
        try:
            single_frame_path = frames[frame_idx]
        except:
            break
        sampled_frames.append(single_frame_path)
    # Pad with last frame if total frames less than num_segments
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[last_frame_id])
    return sampled_frames[:max_segments]


class Qwen3VLReranker():
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        num_frames: int = MAX_FRAMES,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Given a search query, retrieve relevant candidates that answer the query.",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        enable_profiling: bool = True,
        **kwargs,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_profiling = enable_profiling
        self.profile_stats = ProfileStats()

        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.default_instruction = default_instruction

        print(f"[SemanticSearch] Reranker loading with dtype={torch_dtype}, attn={attn_implementation}")
        print(f"[SemanticSearch] Reranker max_pixels={max_pixels}")

        # Handle attention type
        if attn_implementation == "flash_attention_2":
            print("[SemanticSearch] flash_attention_2 not supported, using sdpa")
            attn_implementation = "sdpa"
        elif attn_implementation == "sage":
            print("[SemanticSearch] sage not supported for reranker, using sdpa")
            attn_implementation = "sdpa"

        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **kwargs
        ).to(self.device)

        self.model = lm.model
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True,
            padding_side='left',
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.model.eval()

        token_true_id = self.processor.tokenizer.get_vocab()["yes"]
        token_false_id = self.processor.tokenizer.get_vocab()["no"]
        self.score_linear = self.get_binary_linear(lm, token_true_id, token_false_id)
        self.score_linear.eval()
        self.score_linear.to(self.device).to(self.model.dtype)
        
        print(f"[SemanticSearch] Reranker model loaded on {self.device}")

    def get_binary_linear(self, model, token_yes, token_no):
        lm_head_weights = model.lm_head.weight.data
        weight_yes = lm_head_weights[token_yes]
        weight_no = lm_head_weights[token_no]
        D = weight_yes.size()[0]
        linear_layer = torch.nn.Linear(D, 1, bias=False)
        with torch.no_grad():
            linear_layer.weight[0] = weight_yes - weight_no
        return linear_layer

    @torch.no_grad()
    def compute_scores(self, inputs):
        # Log input shapes to understand the computational load
        if self.enable_profiling and not hasattr(self, '_logged_shapes'):
            self._logged_shapes = True
            print("\n[PROFILE] Input tensor shapes:")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}, dtype={v.dtype}, device={v.device}")
            
            # Calculate approximate FLOPs indicator
            if 'input_ids' in inputs:
                seq_len = inputs['input_ids'].shape[1]
                print(f"  Sequence length: {seq_len} tokens")
            if 'pixel_values' in inputs:
                pv = inputs['pixel_values']
                print(f"  pixel_values total elements: {pv.numel():,}")
                print(f"  pixel_values MB: {pv.numel() * 2 / 1024 / 1024:.2f} MB (bfloat16)")
        
        batch_scores = self.model(**inputs).last_hidden_state[:, -1]
        scores = self.score_linear(batch_scores)
        scores = torch.sigmoid(scores).squeeze(-1).cpu().detach().tolist()
        return scores

    def truncate_tokens_optimized(
        self,
        tokens: List[str],
        max_length: int,
        special_tokens: List[str]
    ) -> List[str]:
        if len(tokens) <= max_length:
            return tokens

        special_tokens_set = set(special_tokens)
        num_special = sum(1 for token in tokens if token in special_tokens_set)
        num_non_special_to_keep = max_length - num_special

        final_tokens = []
        non_special_kept_count = 0
        for token in tokens:
            if token in special_tokens_set:
                final_tokens.append(token)
            elif non_special_kept_count < num_non_special_to_keep:
                final_tokens.append(token)
                non_special_kept_count += 1

        return final_tokens

    def tokenize_single(self, pair, **kwargs):
        """Tokenize a single query-document pair with detailed timing"""
        max_length = self.max_length
        
        # Sub-stage 1a: Apply chat template
        t_template = time.time()
        text = self.processor.apply_chat_template([pair], tokenize=False, add_generation_prompt=True)
        if self.enable_profiling:
            self.profile_stats.record('tokenize.chat_template', time.time() - t_template)
        
        # Sub-stage 1b: Process vision info (THIS IS LIKELY THE BOTTLENECK - loads images from disk)
        t_vision = time.time()
        try:
            images, videos, video_kwargs = process_vision_info(
                [pair], 
                image_patch_size=16, 
                return_video_kwargs=True, 
                return_video_metadata=True
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            videos = None
            video_kwargs = {'do_sample_frames': False}
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}], 
                add_generation_prompt=True, tokenize=False
            )
        if self.enable_profiling:
            self.profile_stats.record('tokenize.process_vision_info', time.time() - t_vision)
        
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        
        # Sub-stage 1c: Processor call (tokenization + image preprocessing)
        t_processor = time.time()
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            truncation=False,
            padding=False,
            **video_kwargs
        )
        if self.enable_profiling:
            self.profile_stats.record('tokenize.processor', time.time() - t_processor)
        
        # Sub-stage 1d: Truncation
        t_truncate = time.time()
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.truncate_tokens_optimized(
                inputs['input_ids'][i][:-5], max_length,
                self.processor.tokenizer.all_special_ids
            ) + inputs['input_ids'][i][-5:]
        if self.enable_profiling:
            self.profile_stats.record('tokenize.truncate', time.time() - t_truncate)
        
        # Sub-stage 1e: Padding
        t_pad = time.time()
        temp_inputs = self.processor.tokenizer.pad(
            {'input_ids': inputs['input_ids']}, 
            padding=True,
            return_tensors="pt", 
            max_length=self.max_length
        )
        for key in temp_inputs:
            inputs[key] = temp_inputs[key]
        if self.enable_profiling:
            self.profile_stats.record('tokenize.pad', time.time() - t_pad)
            
        return inputs

    def format_mm_content(
        self, 
        text, image, video, 
        prefix='Query:', 
        fps=None, max_frames=None,
    ):
        content = []
        content.append({'type': 'text', 'text': prefix})
        
        if not text and not image and not video:
            content.append({'type': 'text', 'text': "NULL"})
            return content

        if video:
            video_content = None
            video_kwargs = {'total_pixels': self.total_pixels}
            if isinstance(video, list):
                video_content = video
                if self.num_frames is not None or self.max_frames is not None:
                    video_content = self._sample_frames(video_content, self.num_frames, self.max_frames)
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele) 
                    for ele in video_content
                ]
            elif isinstance(video, str):
                video_content = video if video.startswith(('http://', 'https://')) else 'file://' + video
                video_kwargs = {'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames}
            else:
                raise TypeError(f"Unrecognized video type: {type(video)}")

            if video_content:
                content.append({
                    'type': 'video', 'video': video_content,
                    **video_kwargs
                })

        if image:
            image_content = None
            if isinstance(image, Image.Image):
                image_content = image
            elif isinstance(image, str):
                image_content = image if image.startswith(('http', 'oss')) else 'file://' + image
            else:
                raise TypeError(f"Unrecognized image type: {type(image)}")

            if image_content:
                content.append({
                    'type': 'image', 'image': image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })

        if text:
            content.append({'type': 'text', 'text': text})
        return content

    def format_mm_instruction(
        self, 
        query_text, query_image, query_video, 
        doc_text, doc_image, doc_video,
        instruction=None, 
        fps=None, max_frames=None
    ):
        inputs = []
        inputs.append({
            "role": "system",
            "content": [{
                "type": "text",
                "text": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            }]
        })
        
        if isinstance(query_text, tuple):
            instruct, query_text = query_text
        else:
            instruct = instruction
            
        contents = []
        contents.append({
            "type": "text",
            "text": '<Instruct>: ' + instruct
        })
        
        query_content = self.format_mm_content(
            query_text, query_image, query_video, prefix='<Query>:', 
            fps=fps, max_frames=max_frames
        )
        contents.extend(query_content)
        
        doc_content = self.format_mm_content(
            doc_text, doc_image, doc_video, prefix='\n<Document>:', 
            fps=fps, max_frames=max_frames
        )
        contents.extend(doc_content)
        
        inputs.append({
            "role": "user",
            "content": contents
        })
        return inputs

    def _log_gpu_stats(self, label: str = ""):
        """Log GPU memory and utilization"""
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[GPU {label}] Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

    def process(
        self,
        inputs,
        batch_size: int = 4,  # Process multiple documents at once
    ) -> list[torch.Tensor]:
        """
        Process query-document pairs and return relevance scores.
        
        Now with detailed profiling to identify bottlenecks!
        """
        instruction = inputs.get('instruction', self.default_instruction)

        query = inputs.get("query", {})
        documents = inputs.get("documents", [])
        if not query or not documents:
            return []

        # Reset profiling stats
        if self.enable_profiling:
            self.profile_stats.reset()
            self._log_gpu_stats("START")

        # Time pair formatting
        t_format_start = time.time()
        pairs = [
            self.format_mm_instruction(
                query.get('text', None),
                query.get('image', None),
                query.get('video', None),
                document.get('text', None),
                document.get('image', None),
                document.get('video', None),
                instruction=instruction,
                fps=inputs.get('fps', self.fps),
                max_frames=inputs.get('max_frames', self.max_frames)
            ) 
            for document in documents
        ]
        if self.enable_profiling:
            self.profile_stats.record('format_pairs', time.time() - t_format_start)

        print(f"[SemanticSearch] Reranking {len(pairs)} documents...")
        print(f"[SemanticSearch] Profiling enabled: detailed timing will be shown")
        t0 = time.time()

        # Process in batches for speed
        final_scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = []
            
            for j, pair in enumerate(batch_pairs):
                doc_start = time.time()
                try:
                    # Stage 1: Tokenization (includes image loading via process_vision_info)
                    t1 = time.time()
                    tokenized = self.tokenize_single(pair)
                    if self.enable_profiling:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        self.profile_stats.record('tokenize_single', time.time() - t1)
                    
                    # Stage 2: Move to device
                    t2 = time.time()
                    tokenized = tokenized.to(self.model.device)
                    if self.enable_profiling:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        self.profile_stats.record('to_device', time.time() - t2)
                    
                    # Stage 3: Model forward pass
                    t3 = time.time()
                    scores = self.compute_scores(tokenized)
                    if self.enable_profiling:
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        self.profile_stats.record('compute_scores', time.time() - t3)
                    
                    batch_scores.extend(scores)
                    
                    if self.enable_profiling:
                        self.profile_stats.record('total_per_doc', time.time() - doc_start)
                    
                except Exception as e:
                    logger.error(f"Error scoring document: {e}")
                    import traceback
                    traceback.print_exc()
                    batch_scores.append(0.0)
            
            final_scores.extend(batch_scores)
            
            # Progress update with timing breakdown
            done = min(i + batch_size, len(pairs))
            elapsed = time.time() - t0
            if done > 0:
                rate = done / elapsed
                eta = (len(pairs) - done) / rate if rate > 0 else 0
                print(f"[SemanticSearch] Reranked {done}/{len(pairs)} ({rate:.2f} docs/sec, ETA: {eta:.1f}s)")

        total_time = time.time() - t0
        print(f"[SemanticSearch] Reranking complete: {len(pairs)} documents in {total_time:.1f}s ({total_time/len(pairs):.2f}s/doc)")
        
        # Print profiling summary
        if self.enable_profiling:
            self._log_gpu_stats("END")
            print(self.profile_stats.summary())
        
        return final_scores
