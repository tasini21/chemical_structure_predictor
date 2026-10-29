import os
import gc
from rdkit.Chem import MolFromSmiles, MolToInchiKey, MolToSmiles
from model import DeNovoLightningModel
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
import torch
import config
from dataloader import CASMIDataModule, get_tokenizer
from collections import defaultdict
import pandas as pd
import numpy as np

COMP_DIR = 'data'

WORKING_DIR = 'working'
RUN_NAME = 'train_llama'
LOGGER_PATH = os.path.join(WORKING_DIR, "lightning_logs")
MAX_EPOCHS = 1
PRECISION = 'bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed'
BATCH_SIZE = 32
SUBMISSION_PATH = os.path.join(WORKING_DIR, 'submission.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(COMP_DIR, 'sample_submission.csv')

class Logger(TensorBoardLogger):
    def log_metrics(self, metrics, step=None):
        # Remove 'epoch' from the metrics dictionary before logging
        metrics_without_epoch = {
            k: v for k, v in metrics.items() if k != "epoch"
        }
        super().log_metrics(metrics_without_epoch, step=step)

logger = Logger(
        save_dir=LOGGER_PATH,
        name=RUN_NAME,
    )

CHECKPOINT_METRIC_TO_MONITOR = 'val_tanimoto'
CHECKPOINT_MONITOR_MODE = 'max'

CHECKPOINT_DIR = os.path.join(WORKING_DIR, 'model_checkpoints')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, RUN_NAME)

checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    dirpath=CHECKPOINT_PATH,
    monitor=CHECKPOINT_METRIC_TO_MONITOR,
    mode=CHECKPOINT_MONITOR_MODE,
    every_n_epochs=1,
    save_top_k=1,
    filename=f"{{epoch}}-{{step}}-{{{CHECKPOINT_METRIC_TO_MONITOR}:.3f}}",
)

if __name__ == '__main__':
    datamodule = CASMIDataModule(batch_size=BATCH_SIZE)
    datamodule.setup('fit')

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        devices=1,
        accelerator='auto',
        precision=PRECISION,
        log_every_n_steps=100,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=3,
        logger=[logger],
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=2,
    )
    model = DeNovoLightningModel(config.model_params)

    trainer.fit(
        model,
        datamodule=datamodule,
        # ckpt_path=RESUME_FROM_CHECKPOINT_PATH,
        weights_only=False,
    )

    PREDICT_BATCH_SIZE = 8
    datamodule.predict_batch_size = PREDICT_BATCH_SIZE
    datamodule.setup('predict')

    gc.collect()
    torch.cuda.empty_cache()

    predict_trainer = L.Trainer(accelerator='auto', devices=1, precision='bf16-mixed', logger=False)
    predictions = predict_trainer.predict(model, dataloaders=datamodule.predict_dataloader())


    N_GUESSES = 25
    FALLBACK_SMILES = 'CCO'  # every molecule_id must carry at least one guess

    best_by_molecule = defaultdict(dict)  # molecule_id -> {inchikey14: (score, smiles)}
    for molecule_ids, token_batch, score_batch in predictions:
        for molecule_id, token_seqs, scores in zip(molecule_ids, token_batch, score_batch):
            for tokens, score in zip(token_seqs, scores):
                decode_tokenized_smiles = get_tokenizer()

                smiles = decode_tokenized_smiles(tokens.tolist())
                mol = MolFromSmiles(smiles) if smiles else None
                if mol is None:
                    continue
                inchikey14 = MolToInchiKey(mol).split('-')[0]
                candidate = (float(score), MolToSmiles(mol))
                if candidate > best_by_molecule[molecule_id].get(inchikey14, (-np.inf, '')):
                    best_by_molecule[molecule_id][inchikey14] = candidate


    def top_guesses(molecule_id):
        ranked = sorted(best_by_molecule.get(molecule_id, {}).values(), reverse=True)
        guesses = [smiles for _, smiles in ranked[:N_GUESSES]]
        return ';'.join(guesses) if guesses else FALLBACK_SMILES


    # Starting from the sample submission guarantees the right molecule_ids, in the right order.
    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission['smiles'] = submission.molecule_id.apply(top_guesses)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f'wrote {len(submission)} rows to {SUBMISSION_PATH}')
    n_guesses = submission.smiles.str.split(';').map(len)
    print(f'guesses per molecule: median {int(n_guesses.median())}, max {n_guesses.max()}')
    submission.head()