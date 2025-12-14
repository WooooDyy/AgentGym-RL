'''
Memory bank implementation for retrieval-based episodic memory in agent training.

This module provides a FAISS-based memory bank that stores and retrieves past experiences
to be used as few-shot examples during agent generation.
'''

import pickle
from dataclasses import dataclass
from pathlib import Path
import faiss
import numpy as np
from numpy.typing import NDArray
import torch
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer


@dataclass
class Experience:
    '''A single experience in the memory bank.'''
    obs_text: str
    action: str
    reward: float
    task_name: str
    item_id: int
    obs_embedding: NDArray | None


class MemoryBank:
    '''
    FAISS-based memory bank for storing and retrieving agent experiences.

    This implementation uses sentence-transformers to encode observations and FAISS
    for efficient similarity search. Experiences are stored with their embeddings,
    enabling fast retrieval of similar past experiences.

    Args:
        encoder_name: Name of the sentence-transformer model to use for encoding
        embedding_dim: Dimension of the embeddings (auto-detected from encoder)
        device: Device to run the encoder on ('cuda' or 'cpu')
        min_reward: Minimum reward threshold for storing experiences
        task_specific: If True, only retrieve from same task; if False, cross-task retrieval
    '''

    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int | None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        min_reward: float = 0.5,
        task_specific: bool = True,
    ) -> None:
        self.encoder_name = encoder_name
        self.device = device
        self.min_reward = min_reward
        self.task_specific = task_specific

        # Initialize encoder
        self.encoder = SentenceTransformer(encoder_name, device=device)
        self.embedding_dim = embedding_dim or self.encoder.get_sentence_embedding_dimension()

        # Initialize FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Store experiences
        self.experiences: list[Experience] = []

        # Task-specific indices for filtering
        self.task_indices: dict[str, list[int]] = {}

    def encode(self, texts: list[str]) -> NDArray:
        '''
        Encode texts into embeddings using the sentence transformer.

        Args:
            texts: List of text strings to encode

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        '''
        with torch.no_grad():
            embeddings = self.encoder.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
            )
        return embeddings.astype('float32')

    def add(
        self,
        obs_text: str,
        action: str,
        reward: float,
        task_name: str,
        item_id: int,
    ) -> bool:
        '''
        Add a new experience to the memory bank.

        Args:
            obs_text: Observation text (will be encoded)
            action: Action taken
            reward: Reward received
            task_name: Name of the task
            item_id: Item ID for the task instance

        Returns:
            True if experience was added, False if filtered out by min_reward
        '''
        # Filter by reward threshold
        if reward < self.min_reward:
            return False

        # Encode observation
        obs_embedding = self.encode([obs_text])[0]

        # Create experience
        exp = Experience(
            obs_text=obs_text,
            action=action,
            reward=reward,
            task_name=task_name,
            item_id=item_id,
            obs_embedding=obs_embedding,
        )

        # Add to index
        self.index.add(obs_embedding.reshape(1, -1))

        # Store experience
        exp_idx = len(self.experiences)
        self.experiences.append(exp)

        # Update task indices
        if task_name not in self.task_indices:
            self.task_indices[task_name] = []
        self.task_indices[task_name].append(exp_idx)

        return True

    def retrieve(
        self,
        query_text: str,
        k: int = 3,
        task_name: str | None,
    ) -> list[Experience]:
        '''
        Retrieve top-k similar experiences for a query observation.

        Args:
            query_text: Query observation text
            k: Number of experiences to retrieve
            task_name: Task name for filtering (required if task_specific=True)

        Returns:
            List of top-k most similar experiences, sorted by similarity (most similar first)
        '''
        if len(self.experiences) == 0:
            return []

        # Handle task-specific retrieval
        if self.task_specific:
            if task_name is None:
                raise ValueError("task_name required for task-specific retrieval")

            # Get indices for this task
            valid_indices = self.task_indices.get(task_name, [])
            if len(valid_indices) == 0:
                return []

            # Limit k to available experiences
            k = min(k, len(valid_indices))
        else:
            valid_indices = list(range(len(self.experiences)))
            k = min(k, len(valid_indices))

        # Encode query
        query_embedding = self.encode([query_text])[0].reshape(1, -1)

        if self.task_specific and len(valid_indices) < len(self.experiences):
            # Need to search among filtered experiences
            # Build temporary index for this task
            task_embeddings_list = [
                self.experiences[idx].obs_embedding for idx in valid_indices
            ]
            # Filter out None values (should not happen, but for type safety)
            task_embeddings_list = [e for e in task_embeddings_list if e is not None]
            task_embeddings = np.vstack(task_embeddings_list)
            temp_index = faiss.IndexFlatL2(self.embedding_dim)
            temp_index.add(task_embeddings)

            # Search
            _, indices = temp_index.search(query_embedding, k)

            # Map back to global indices
            retrieved_indices = [valid_indices[idx] for idx in indices[0]]
        else:
            # Search full index
            distances, indices = self.index.search(query_embedding, k)
            retrieved_indices = indices[0].tolist()

        # Return experiences
        return [self.experiences[idx] for idx in retrieved_indices]

    def format_as_examples(
        self,
        experiences: list[Experience],
        tokenizer: PreTrainedTokenizer = None,
        format_style: str = "chat",
    ) -> list[dict]:
        '''
        Format retrieved experiences as few-shot examples for chat templates.

        Args:
            experiences: List of experiences to format
            tokenizer: Tokenizer for potential formatting adjustments
            format_style: Format style ('chat' for chat template format)

        Returns:
            List of formatted message dictionaries
        '''
        if format_style == "chat":
            examples = []
            for exp in experiences:
                # Add user message (observation)
                examples.append({
                    "role": "user",
                    "content": exp.obs_text
                })
                # Add assistant message (action)
                examples.append({
                    "role": "assistant",
                    "content": exp.action
                })
            return examples
        else:
            raise ValueError(f"Unsupported format_style: {format_style}")

    def save(self, save_path: str) -> None:
        '''
        Save memory bank to disk.

        Args:
            save_path: Path to save the memory bank
        '''
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(save_path_obj.with_suffix('.faiss')))

        # Save experiences and metadata
        state = {
            'experiences': self.experiences,
            'task_indices': self.task_indices,
            'encoder_name': self.encoder_name,
            'embedding_dim': self.embedding_dim,
            'min_reward': self.min_reward,
            'task_specific': self.task_specific,
        }
        with open(save_path_obj.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, load_path: str, device: str | None) -> "MemoryBank":
        '''
        Load memory bank from disk.

        Args:
            load_path: Path to load the memory bank from
            device: Device to load the encoder on (defaults to saved device)

        Returns:
            Loaded MemoryBank instance
        '''
        load_path_obj = Path(load_path)

        # Load experiences and metadata
        with open(load_path_obj.with_suffix('.pkl'), 'rb') as f:
            state = pickle.load(f)

        # Create instance
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        memory_bank = cls(
            encoder_name=state['encoder_name'],
            embedding_dim=state['embedding_dim'],
            device=device,
            min_reward=state['min_reward'],
            task_specific=state['task_specific'],
        )

        # Load FAISS index
        memory_bank.index = faiss.read_index(str(load_path_obj.with_suffix('.faiss')))

        # Restore experiences and task indices
        memory_bank.experiences = state['experiences']
        memory_bank.task_indices = state['task_indices']

        return memory_bank

    def __len__(self) -> int:
        '''Return number of experiences in the memory bank.'''
        return len(self.experiences)

    def clear(self) -> None:
        '''Clear all experiences from the memory bank.'''
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.experiences = []
        self.task_indices = {}
