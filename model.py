import torch
import torch.nn as nn
from rdkit import DataStructs
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.nn.functional as F
from torch.optim import AdamW
from dataloader import get_tokenizer
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import lightning as L
from rdkit.Chem import MolFromSmiles, MolToInchiKey, rdFingerprintGenerator

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
def tanimoto_smiles(smiles1, smiles2):
    mol1, mol2 = MolFromSmiles(smiles1), MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None
    fp1, fp2 = fp_gen.GetSparseFingerprint(mol1), fp_gen.GetSparseFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

class MetadataEmbedder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, metadata):
        return self.encoder(metadata)

### peak embedder
class PeakEmbedder(nn.Module):
    """embed (m/z, intensity) peak pairs with sinusoidal embeddings from Voronov et al"""

    def __init__(self, d_model, dropout, sin_dim=None, mz_log_lims=(-2., 3.), mz_log_power=1.0):
        super().__init__()
        sin_dim = sin_dim if sin_dim is not None else d_model
        self.dropout = dropout

        wavelength = torch.pow(
            10,
            (mz_log_lims[1] - mz_log_lims[0]) * torch.pow(
                torch.linspace(0, 1, int(sin_dim / 2)),
                mz_log_power,
            ) + mz_log_lims[0],
        )
        frequency = 2 * np.pi / wavelength
        self._frequency = nn.Parameter(frequency, requires_grad=False)
        self._ff_block_1 = nn.Sequential(*[
            nn.Linear(sin_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        ])
        self._ff_block_2 = nn.Sequential(*[
            nn.Linear(d_model + 1, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        ])

    def forward(self, mz_tensor, intensity_tensor):
        # "mz embedding" embeds just the m/z value
        omega_mz = self._frequency.view(
            *(1 for _ in range(mz_tensor.ndim)), -1
        ) * mz_tensor.unsqueeze(-1)
        sin = torch.sin(omega_mz)
        cos = torch.cos(omega_mz)
        mz_vecs = torch.cat([sin, cos], dim=-1)
        mz_embeds = self._ff_block_1(mz_vecs)

        # "peak embedding" embeds both the m/z and intensity
        peak_embeds = torch.cat([mz_embeds, intensity_tensor.unsqueeze(2)], dim=2)
        peak_embeds = self._ff_block_2(peak_embeds)
        return peak_embeds

### spectrum encoder
class SpectrumEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers, dim_feedforward=None, dropout=0.1, activation='relu'):
        super().__init__()
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * embed_dim
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                embed_dim,
                n_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                dropout=dropout,
                activation=activation,
            ),
            n_layers,
        )
        self.init_weights()

    def forward(self, sequence_input, attention_mask):
        pad_mask = (attention_mask == 0)
        return self.encoder(sequence_input, src_key_padding_mask=pad_mask)

    def init_weights(self):
        for layer in self.encoder.layers:
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)
                elif isinstance(mod, nn.LayerNorm):
                    nn.init.constant_(mod.weight, 1.0)
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)

### smiles decoder
class LlamaSmilesDecoder(nn.Module):
    def __init__(
            self,
            encoder_dim,
            model_name="meta-llama/Llama-3.2-1B",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"),
            max_smiles_tokens=128,
            torch_dtype=torch.bfloat16,
            validate_n_samples=10,
            predict_n_samples=25,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_token_id
        self.max_smiles_tokens = max_smiles_tokens
        self.n_samples_per_stage = {'validate': validate_n_samples, 'predict': predict_n_samples}

        llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(llm, lora_cfg)  # freezes base weights, adds adapters
        hidden = llm.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.prefix_norm = nn.LayerNorm(hidden)

    # ---------- helpers ----------
    def _embed(self, ids):
        return self.llm.get_input_embeddings()(ids)

    def _project(self, encoder_outputs):
        prefix = self.prefix_norm(self.projector(encoder_outputs))
        return prefix.to(self.llm.get_input_embeddings().weight.dtype)

    @staticmethod
    def _position_ids(attn):
        # padded peaks sit in the middle of the sequence, so derive positions from the mask
        return (attn.cumsum(-1) - 1).clamp(min=0)

    def _token_mask(self, idx):
        # real tokens = everything up to and including the first EOS (position 0 is BOS)
        is_eos = (idx == self.eos_token_id).long()
        is_eos[:, 0] = 0
        return (is_eos.cumsum(-1) - is_eos) == 0

    def tokenize(self, smiles, device=None):
        """list[str] -> (B, T) LongTensor: [BOS] smiles [EOS] [PAD...]"""
        ids = self.tokenizer(list(smiles), add_special_tokens=False)["input_ids"]
        ids = [[self.bos_token_id] + x[: self.max_smiles_tokens] + [self.eos_token_id] for x in ids]
        T = max(len(x) for x in ids)
        out = torch.full((len(ids), T), self.pad_token_id, dtype=torch.long)
        for i, x in enumerate(ids):
            out[i, : len(x)] = torch.tensor(x)
        return out.to(device) if device is not None else out

    def decode(self, token_ids):
        """list[int] (starting with BOS) -> SMILES string"""
        ids = list(token_ids)[1:]
        if self.eos_token_id in ids:
            ids = ids[: ids.index(self.eos_token_id)]
        return self.tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "").strip()

    # ---------- same signature as before ----------
    def forward(self, idx, encoder_outputs, encoder_attention_mask, structure_tokens=None):
        S = encoder_outputs.shape[1]
        tok_mask = self._token_mask(idx)
        inputs_embeds = torch.cat([self._project(encoder_outputs), self._embed(idx)], dim=1)
        attn = torch.cat([encoder_attention_mask.long(), tok_mask.long()], dim=1)

        x = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            position_ids=self._position_ids(attn),
        )
        logits = x.logits[:, S:].float()  # keep only the SMILES positions

        output_dict = {'logits': logits}
        if structure_tokens is not None:
            # logits at t predict token t+1 (the LLM is a standard causal LM)
            labels = structure_tokens[:, 1:].masked_fill(~tok_mask[:, 1:], -100)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction='mean',
            )
            output_dict['loss'] = loss
        return output_dict

    @torch.inference_mode()
    def generate(
            self,
            encoder_states,
            encoder_mask,
            n_samples=1,
            max_new_tokens=50,
            temperature=1.0,
    ):
        """sampling based generation; returns (B, n_samples, L) token ids (starting with BOS) and (B, n_samples) log-probs"""
        B = encoder_states.shape[0]
        dev = encoder_states.device
        prefix = self._project(encoder_states)
        bos = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=dev)
        inputs_embeds = torch.cat([prefix, self._embed(bos)], dim=1)
        attn = torch.cat([encoder_mask.long(), torch.ones(B, 1, dtype=torch.long, device=dev)], dim=1)

        inputs_embeds = inputs_embeds.repeat_interleave(n_samples, dim=0)
        attn = attn.repeat_interleave(n_samples, dim=0)

        out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        seq = out.sequences  # (B*n, T), new tokens only
        logp = torch.stack(out.scores, dim=1).log_softmax(-1)  # (B*n, T, V)
        tok_logp = logp.gather(-1, seq.unsqueeze(-1)).squeeze(-1)

        is_eos = (seq == self.eos_token_id).long()
        after_eos = (is_eos.cumsum(-1) - is_eos) > 0  # tokens after first EOS
        tok_logp = tok_logp.masked_fill(after_eos, 0.0)

        idx = torch.cat([bos.repeat_interleave(n_samples, dim=0), seq], dim=1)
        _, Lq = idx.shape
        return idx.reshape(B, n_samples, Lq), tok_logp.sum(dim=-1).reshape(B, n_samples)

class DeNovoLightningModel(L.LightningModule):
    def __init__(self, kwargs):
        super().__init__()
        self.peak_embedder = PeakEmbedder(**kwargs['peak_embedder'])
        self.metadata_embedder = MetadataEmbedder(**kwargs['metadata_embedder'])
        self.spectrum_encoder = SpectrumEncoder(**kwargs['spectrum_encoder'])
        self.smiles_decoder = LlamaSmilesDecoder(**kwargs['smiles_decoder'])
        self.optimizer_config = kwargs['optimizer']

    def forward(self, batch, stage, return_encoder_states=False):
        peak_embeds = self.peak_embedder(
            batch["mzs"],
            batch["intensities"]
        )

        metadata_embeds = self.metadata_embedder(
            batch["metadata"]
        )

        peak_embeds += metadata_embeds.unsqueeze(1)

        encoded_spectra = self.spectrum_encoder(
            peak_embeds,
            batch["attention_mask"]
        )
        outputs = {}
        if stage in ('fit', 'validate'):
            outputs = self.smiles_decoder(
                batch['structure_tokens'],
                encoded_spectra,
                batch['attention_mask'],
                structure_tokens=batch['structure_tokens']
            )
        if stage in ('validate', 'predict'):
            tokens, scores = self.smiles_decoder.generate(
                encoded_spectra,
                batch['attention_mask'],
                n_samples=self.smiles_decoder.n_samples_per_stage[stage],
                temperature=1., # experiment with this
            )
            outputs['generated_tokens'] = tokens
            outputs['generated_scores'] = scores
        if return_encoder_states:
            outputs['encoder_states'] = encoded_spectra
        return outputs

    def training_step(self, batch):
        outputs = self(batch, "fit")
        loss = outputs['loss']
        self.log(
            "train_loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=batch['mzs'].shape[0],
        )
        return loss

    def validation_step(self, batch):
        forward_outputs = self(batch, "validate", return_encoder_states=True)
        loss = forward_outputs['loss']
        generated_tokens, generated_scores = self.smiles_decoder.generate(
            forward_outputs['encoder_states'],
            batch['attention_mask'],
            n_samples=10,
            temperature=1.0,
        )
        generation_metrics = self.compute_val_metrics(
            generated_tokens,
            generated_scores,
            batch['smiles']
        )

        metrics_dict = {
            "val_loss": loss,
            **{f"val_{k}": v for k, v in generation_metrics.items()},
        }
        self.log_dict(
            metrics_dict,
            add_dataloader_idx=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=(batch['mzs'].shape[0]),
        )

    def predict_step(self, batch):
        # forward() already generates for stage='predict'; generating a second time here just did
        # the same expensive sampling twice.
        outputs = self(batch, "predict")
        return batch['molecule_id'], outputs['generated_tokens'], outputs['generated_scores']

    def compute_val_metrics(self, generated_tokens, generated_scores, labels):
        decode_tokenized_smiles = get_tokenizer()
        decoded_smiles = [
            [decode_tokenized_smiles(tokens.tolist()) for tokens in token_seqs]
            for token_seqs in generated_tokens
        ]
        score_masks = [scores.argsort(descending=True) for scores in generated_scores]
        sorted_valid_smiles = [
            [smiles_seq[idx] for idx in mask if smiles_seq[idx] != '' and MolFromSmiles(smiles_seq[idx]) is not None]
            for smiles_seq, mask in zip(decoded_smiles, score_masks)
        ]
        top_valid_smiles = [
            smiles_seq[0]
            if len(smiles_seq) > 0 else None
            for smiles_seq in sorted_valid_smiles
        ]

        def _sm_to_ikey(smiles):
            return MolToInchiKey(MolFromSmiles(smiles)).split('-')[0]
        ikey_match = [
            _sm_to_ikey(smiles) == _sm_to_ikey(label)
            if smiles is not None else False
            for smiles, label in zip(top_valid_smiles, labels)
        ]
        tanimotos = [
            tanimoto_smiles(smiles, label)
            if smiles is not None else 0.0
            for smiles, label in zip(top_valid_smiles, labels)
        ]

        frac_ikey_match = np.mean(ikey_match)
        frac_valid_smiles = np.mean([x is not None for x in top_valid_smiles])
        mean_tanimoto = np.mean(tanimotos)

        return {
            'valid_smiles': frac_valid_smiles,
            'tanimoto': mean_tanimoto,
            'exact match': frac_ikey_match,
        }

    def configure_optimizers(self):
        return AdamW(
            self.parameters(),
            lr=self.optimizer_config['lr'],
        )