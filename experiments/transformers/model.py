import torch
from transfer_nlp.plugins.config import register_plugin
from pytorch_pretrained_bert import BertTokenizer


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
