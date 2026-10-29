# model hyperparams
EMBED_DIM = 768
DIM_FEEDFORWARD = 3072
DROPOUT = 0.1

ENCODER_N_HEADS = 12
ENCODER_N_LAYERS = 6
ENCODER_ACTIVATION = 'gelu'

# LLM decoder (replaces DECODER_N_HEADS / N_LAYERS / ACTIVATION)
LLM_NAME = "meta-llama/Llama-3.2-1B"   # or 'Qwen/Qwen2.5-0.5B' if you don't want a gated model
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SMILES_TOKENS = 128

N_SAMPLES_VAL = 10
N_SAMPLES_PRED = 25

LEARNING_RATE = 5e-5      # spectrum encoder, embedders, projector
LORA_LEARNING_RATE = 2e-4 # LoRA adapters

model_params = {
    'peak_embedder': {
        'd_model': EMBED_DIM,
        'dropout': DROPOUT,
    },
    'metadata_embedder': {
        'd_model': EMBED_DIM,
    },
    'spectrum_encoder': {
        'embed_dim': EMBED_DIM,
        'n_heads': ENCODER_N_HEADS,
        'n_layers': ENCODER_N_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT,
        'activation': ENCODER_ACTIVATION,
    },
    'smiles_decoder': {
        'encoder_dim': EMBED_DIM,
        'model_name': LLM_NAME,
        'lora_r': LORA_R,
        'lora_alpha': LORA_ALPHA,
        'lora_dropout': LORA_DROPOUT,
        'max_smiles_tokens': MAX_SMILES_TOKENS,
        'validate_n_samples': N_SAMPLES_VAL,
        'predict_n_samples': N_SAMPLES_PRED,
    },
    'optimizer': {
        'lr': LEARNING_RATE,
        'lora_lr': LORA_LEARNING_RATE,
    },
}
