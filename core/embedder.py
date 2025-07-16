# %%
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

class Embedder:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, texts, cache_path=None, force_recompute=False):
        if cache_path and os.path.exists(cache_path) and not force_recompute:
            return pd.read_csv(cache_path)

        embeddings = self.model.encode(texts, show_progress_bar=True, device=self.device)
        df = pd.DataFrame(embeddings)

        if cache_path:
            df.to_csv(cache_path, index=False)

        return df

    def embed_with_hidden_states(self, texts, return_numpy=True):
        """
        Returns a list of hidden states (one tensor per layer) for each input sentence.
        If return_numpy is True, returns numpy arrays instead of torch tensors.
        Output shape: list of [num_layers][batch_size, seq_len, hidden_dim]
        """
        # Access the underlying HuggingFace model and tokenizer
        tokenizer = self.model.tokenizer
        model = self.model._first_module().auto_model
        # Tokenize
        encoded = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, return_dict=True)
        # outputs.hidden_states: tuple of (layer, batch, seq, hidden)
        # Transpose to [layer][batch, seq, hidden]
        hidden_states = outputs.hidden_states
        if return_numpy:
            hidden_states = [h.cpu().numpy() for h in hidden_states]
        return hidden_states

    def embed_with_meanpooled_hidden_states(self, texts, cache_path=None, force_recompute=False, return_numpy=True, batch_size=32):
        """
        Returns a list of mean pooled hidden states (one vector per layer per sentence).
        If cache_path is provided, saves/loads a path/to/cache.npz file with all layers.
        Output shape: [num_layers][batch_size, hidden_dim]
        """
        if cache_path and os.path.exists(cache_path) and not force_recompute:
            loaded = np.load(cache_path, allow_pickle=True)
            mean_pooled = [loaded[f'layer_{i}'] for i in range(len(loaded.files))]
            return mean_pooled

        tokenizer = self.model.tokenizer
        model = self.model._first_module().auto_model
        model.to(self.device)
        model.eval()

        all_layer_outputs = None  # List of layer outputs to be built up across batches

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            input_mask = encoded["attention_mask"].to(self.device)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded, output_hidden_states=True, return_dict=True)

            hidden_states = outputs.hidden_states  # tuple: (num_layers, batch, seq, hidden)
            batch_means = []
            for layer in hidden_states:
                mask = input_mask.unsqueeze(-1).expand(layer.size())
                layer = layer * mask
                summed = layer.sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                mean = summed / counts
                batch_means.append(mean.cpu().numpy() if return_numpy else mean)

            if all_layer_outputs is None:
                all_layer_outputs = [[] for _ in range(len(batch_means))]
            for i, mean in enumerate(batch_means):
                all_layer_outputs[i].append(mean)

        mean_pooled = [np.concatenate(layer, axis=0) if return_numpy else torch.cat(layer, dim=0) for layer in all_layer_outputs]

        if cache_path:
            np.savez(cache_path, **{f'layer_{i}': arr for i, arr in enumerate(mean_pooled)})

        return mean_pooled

