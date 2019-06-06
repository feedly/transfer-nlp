"""
This file contains models presented in the Transfer Learning for NLP Tutorial at NAACL 2019
Models are adapted from https://colab.research.google.com/drive/1iDHCYIrWswIKp-n-pOg69xLoZO09MEgf#scrollTo=_FfRT6GTjHhC&forceEdit=true&offline=true&sandboxMode=true

This is a WIP document and work is needed so that we don't have to replicate so many transformer classes
Ideally we'd like to have flexible transformer classes from which we can easily add
task-dependent heads and add adapter tools, e.g. freezing the backbone and add
residual connexion between layers. 
"""

import torch
from pytorch_pretrained_bert import BertTokenizer

from transfer_nlp.plugins.config import register_plugin


@register_plugin
class Transformer(torch.nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_embeddings: int, num_max_positions: int, num_heads: int, num_layers: int, dropout: float,
                 causal: bool):
        super().__init__()
        self.causal: bool = causal
        self.tokens_embeddings: torch.nn.Embedding = torch.nn.Embedding(num_embeddings, embed_dim)
        self.position_embeddings: torch.nn.Embedding = torch.nn.Embedding(num_max_positions, embed_dim)
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(dropout)

        self.attentions, self.feed_forwards = torch.nn.ModuleList(), torch.nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = torch.nn.ModuleList(), torch.nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(torch.nn.Sequential(torch.nn.Linear(embed_dim, hidden_dim),
                                                          torch.nn.ReLU(),
                                                          torch.nn.Linear(hidden_dim, embed_dim)))
            self.layer_norms_1.append(torch.nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(torch.nn.LayerNorm(embed_dim, eps=1e-12))

        self.attn_mask = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    def forward(self, x):
        """ x has shape [batch, seq length]"""

        padding_mask = (x == self.tokenizer.vocab['[PAD]'])

        x = x.transpose(0, 1).contiguous()

        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)

        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms_1, self.attentions,
                                                                       self.layer_norms_2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False, key_padding_mask=padding_mask)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h


@register_plugin
class TransformerWithLMHead(torch.nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_max_positions: int, num_heads: int, num_layers: int, dropout: float, causal: bool,
                 initializer_range: float):
        """ Transformer with a language modeling head on top (tied weights) """
        super().__init__()
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        num_embeddings = len(tokenizer.vocab)
        self.initializer_range = initializer_range
        self.transformer = Transformer(embed_dim, hidden_dim, num_embeddings,
                                       num_max_positions, num_heads, num_layers,
                                       dropout, causal=causal)

        self.lm_head = torch.nn.Linear(embed_dim, num_embeddings, bias=False)
        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.transformer.tokens_embeddings.weight

    def init_weights(self, module):
        """ initialize weights - nn.MultiheadAttention is already initalized by PyTorch (xavier) """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        """ x has shape [batch, seq length]"""
        hidden_states = self.transformer(x)
        logits = self.lm_head(hidden_states)

        return logits


@register_plugin
class LMLoss:

    def __init__(self, causal: bool):
        self.causal: bool = causal

    def __call__(self, input, target):
        input = input.transpose(0, 1).contiguous()
        shift_logits = input[:-1] if self.causal else input
        shift_labels = target[1:] if self.causal else target
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss


@register_plugin
class TransformerWithClfHead(torch.nn.Module):
    def __init__(self,
                 embed_dim: int, hidden_dim: int, num_max_positions: int, num_heads: int, num_layers: int, dropout: float, causal: bool,
                 initializer_range: float, num_classes: int):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        num_embeddings = len(self.tokenizer.vocab)
        self.initializer_range = initializer_range
        self.transformer = Transformer(embed_dim, hidden_dim, num_embeddings,
                                       num_max_positions, num_heads, num_layers,
                                       dropout, causal=causal)

        self.classification_head = torch.nn.Linear(embed_dim, num_classes)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):

        # x = x.transpose(0, 1).contiguous().to('cpu')
        clf_tokens_mask = (x.transpose(0, 1).contiguous().to('cpu') == self.tokenizer.vocab['[CLS]'])

        hidden_states = self.transformer(x)
        msk = clf_tokens_mask.unsqueeze(-1).float()
        clf_tokens_states = (hidden_states * msk).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)

        return clf_logits


@register_plugin
class FineTuningLoss:

    def __call__(self, input, target):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(input.view(-1, input.size(-1)), target.view(-1))
        return loss


class TransformerWithAdapters(Transformer):
    def __init__(self, adapters_dim, embed_dim, hidden_dim, num_embeddings, num_max_positions,
                 num_heads, num_layers, dropout, causal):
        """ Transformer with adapters (small bottleneck layers) """
        super().__init__(embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers,
                         dropout, causal)
        self.adapters_1 = torch.nn.ModuleList()
        self.adapters_2 = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.adapters_1.append(torch.nn.Sequential(torch.nn.Linear(embed_dim, adapters_dim),
                                                       torch.nn.ReLU(),
                                                       torch.nn.Linear(adapters_dim, embed_dim)))

            self.adapters_2.append(torch.nn.Sequential(torch.nn.Linear(embed_dim, adapters_dim),
                                                       torch.nn.ReLU(),
                                                       torch.nn.Linear(adapters_dim, embed_dim)))

    def forward(self, x):
        """ x has shape [batch, seq length]"""

        padding_mask = (x == self.tokenizer.vocab['[PAD]'])

        x = x.transpose(0, 1).contiguous()

        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)

        for (layer_norm_1, attention, adapter_1, layer_norm_2, feed_forward, adapter_2) \
                in zip(self.layer_norms_1, self.attentions, self.adapters_1,
                       self.layer_norms_2, self.feed_forwards, self.adapters_2):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False, key_padding_mask=padding_mask)
            x = self.dropout(x)

            x = adapter_1(x) + x  # Add an adapter with a skip-connection after attention module

            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)

            x = adapter_2(x) + x  # Add an adapter with a skip-connection after feed-forward module

            h = x + h
        return h


@register_plugin
class TransformerWithClfHeadAndAdapters(torch.nn.Module):
    def __init__(self, adapters_dim: int,
                 embed_dim: int, hidden_dim: int, num_max_positions: int, num_heads: int, num_layers: int, dropout: float, causal: bool,
                 initializer_range: float, num_classes: int):
        """ Transformer with a classification head and adapters. """
        super().__init__()
        self.initializer_range: float = initializer_range
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        num_embeddings = len(self.tokenizer.vocab)
        self.num_layers = num_layers
        self.transformer: TransformerWithAdapters = TransformerWithAdapters(adapters_dim, embed_dim, hidden_dim, num_embeddings,
                                       num_max_positions, num_heads, num_layers,
                                       dropout, causal=causal)

        self.classification_head = torch.nn.Linear(embed_dim, num_classes)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):

        clf_tokens_mask = (x.transpose(0, 1).contiguous().to('cpu') == self.tokenizer.vocab['[CLS]'])
        hidden_states = self.transformer(x)
        clf_tokens_states = (hidden_states * clf_tokens_mask.unsqueeze(-1).float()).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)

        return clf_logits


@register_plugin
class TransformerWithClfHeadAndLMHead(torch.nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_max_positions: int, num_heads: int, num_layers: int, dropout: float, causal: bool,
                 initializer_range: float, num_classes: int):
        super().__init__()
        self.initializer_range: float = initializer_range
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        num_embeddings = len(self.tokenizer.vocab)
        self.num_layers = num_layers
        self.transformer = Transformer(embed_dim, hidden_dim, num_embeddings,
                                       num_max_positions, num_heads, num_layers,
                                       dropout, causal=causal)

        self.lm_head = torch.nn.Linear(embed_dim, num_embeddings, bias=False)
        self.classification_head = torch.nn.Linear(embed_dim, num_classes)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.transformer.tokens_embeddings.weight

    def init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        """ x and clf_tokens_mask have shape [seq length, batch] padding_mask has shape [batch, seq length] """
        clf_tokens_mask = (x.transpose(0, 1).contiguous().to('cpu') == self.tokenizer.vocab['[CLS]'])
        hidden_states = self.transformer(x)

        lm_logits = self.lm_head(hidden_states)
        clf_tokens_states = (hidden_states * clf_tokens_mask.unsqueeze(-1).float()).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)

        return lm_logits, clf_logits
    
@register_plugin
class MultiTaskLoss:
    
    def __init__(self, causal: bool):
        self.causal: bool = causal

    def __call__(self, lm_logits, clf_logits, lm_labels, clf_labels):
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss_clf = loss_fct(clf_logits.view(-1, clf_logits.size(-1)), clf_labels.view(-1))

        shift_logits = lm_logits[:-1] if self.causal else lm_logits
        shift_labels = lm_labels[1:] if self.causal else lm_labels
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        loss_lm = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss_lm, loss_clf