import os
from functools import partial
import rdkit.RDLogger as rkl
import rdkit.rdBase as rkrb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tokenizers import Tokenizer, models, trainers
from tokenizers.processors import TemplateProcessing
from torch.utils.data import Dataset, DataLoader
import torch
import lightning as L
from tqdm import tqdm
tqdm.pandas()

COMP_DIR = 'data'
TRAIN_PATH = os.path.join(COMP_DIR, 'train.parquet')
TEST_PATH = os.path.join(COMP_DIR, 'test.parquet')
SAMPLE_SUBMISSION_PATH = os.path.join(COMP_DIR, 'sample_submission.csv')

WORKING_DIR = 'working'
SUBMISSION_PATH = os.path.join(WORKING_DIR, 'submission.csv')

logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")

NEEDED = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'precursor_mz',
          'ms2_mzs', 'ms2_normalized_intensities', 'precursor_mz', 'adduct',
          'ionization_mode', 'collision_energy_ev']
casmi_all_df = pd.read_parquet(TRAIN_PATH, columns=NEEDED,
                             filters=[('ingest_lib', '!=', 'enveda-180')])

all_structs = casmi_all_df.inchikey14.unique().to_numpy()
train_structs, val_structs = train_test_split(all_structs, test_size=500, random_state=0)
casmi_train_df = casmi_all_df[casmi_all_df.inchikey14.isin(train_structs)]
casmi_val_df = casmi_all_df[casmi_all_df.inchikey14.isin(val_structs)]
casmi_test_df = pd.read_parquet(TEST_PATH)

MAX_TRAIN_SPECTRA = 200_000
if MAX_TRAIN_SPECTRA is not None and len(casmi_train_df) > MAX_TRAIN_SPECTRA:
    casmi_train_df = casmi_train_df.sample(n=MAX_TRAIN_SPECTRA, random_state=0)

casmi_val_df = casmi_val_df.drop_duplicates(subset=['inchikey14'], keep='first')

MAX_LEN = 128
BPE_PAD_ID = 0
BPE_BOS_ID = 1
BPE_EOS_ID = 2
VOCAB_SIZE = 512


# process data
## spectrum transform
def process_spectra(df):
    def _process_spectrum(row):
        # sort mzs, intensities
        sort_mask = np.argsort(row.ms2_normalized_intensities)[::-1]
        sorted_mzs = row.ms2_mzs[sort_mask]
        sorted_intensities = row.ms2_normalized_intensities[sort_mask]

        # truncate to max len
        truncated_mzs = sorted_mzs[
            :MAX_LEN - 1]  # note: we use MAX_LEN - 1 to account for the precursor, which will be prepended
        truncated_intensities = sorted_intensities[:MAX_LEN - 1]

        # normalize intensities
        normalized_intensities = truncated_intensities / truncated_intensities.max()

        return truncated_mzs, normalized_intensities

    mzs, ints = zip(*df.progress_apply(_process_spectrum, axis=1))
    df['processed_mzs'] = mzs
    df['processed_intensities'] = ints
    return df

def train_bpe_tokenizer(structures):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # turn on tokenizers parallelism to speed up bpe training

    full_alphabet = list(set(list("".join(structures))))
    tokenizer = Tokenizer(models.BPE())
    special_tokens = [
        ("<pad>", BPE_PAD_ID),
        ("<s>", BPE_BOS_ID),
        ("</s>", BPE_EOS_ID),
    ]
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=special_tokens
    )
    special_tokens = ["<pad>", "<s>", "</s>"]

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        initial_alphabet=full_alphabet,
        special_tokens=special_tokens,
        show_progress=True,
    )
    train_iterator = structures
    tokenizer.train_from_iterator(train_iterator, trainer)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # turn off tokenizers parallelism
    return tokenizer


def tokenize_df_smiles(df, tokenizer):
    df['bpe_tokenized_smiles'] = df.normalized_smiles.progress_apply(
        lambda x: np.array(tokenizer.encode(x).ids, dtype=int)
    )
    return df

casmi_train_df = process_spectra(casmi_train_df)
casmi_val_df = process_spectra(casmi_val_df)
casmi_test_df = process_spectra(casmi_test_df)

## tokenize smiles
print('training bpe...')
bpe_train_structures = list(casmi_train_df.normalized_smiles.unique())
tokenizer = train_bpe_tokenizer(bpe_train_structures)
print('processing smiles...')
casmi_train_df = tokenize_df_smiles(casmi_train_df, tokenizer=tokenizer)
casmi_val_df = tokenize_df_smiles(casmi_val_df, tokenizer=tokenizer)

# utility for decoding bpe smiles
def decode_tokenized_smiles(bpe_smiles_tokens, tokenizer=tokenizer):
    return tokenizer.decode(bpe_smiles_tokens).replace(' ', '')

def get_tokenizer():
    return decode_tokenized_smiles

PRECURSOR_INTENSITY = 2.0
MZ_PAD_ID = 0
INTENSITY_PAD_ID = 0
NUM_WORKERS = os.cpu_count() - 1

class PandasDataset(Dataset):
    def __init__(self, df, column_list, shuffle=True):
        # Prediction keeps the original row order so rows stay aligned with their molecule_id.
        self.df = df[column_list].sample(frac=1, random_state=0) if shuffle else df[column_list]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()


# define datamodule class
ADDUCT_MAP = {
    "[M+H]+": 0,
    "[M-H]-": 1,
    "[M+Na]+": 2,
    "[M+K]+": 3,
}

ION_MODE_MAP = {
    "positive": 0,
    "negative": 1,
}


class CASMIDataModule(L.LightningDataModule):
    def __init__(self, batch_size, predict_batch_size=None):
        super().__init__()
        self.batch_size = batch_size
        # Generation expands the batch by n_samples, so prediction needs a far smaller batch than
        # training. See the predict cell for the arithmetic.
        self.predict_batch_size = predict_batch_size or batch_size

    def setup(self, stage):
        columns = ['precursor_mz', 'processed_mzs', 'processed_intensities', 'bpe_tokenized_smiles',
                   'normalized_smiles']
        if stage == "fit":
            self.train_df = PandasDataset(casmi_train_df, column_list=columns)
            self.val_df = PandasDataset(casmi_val_df, column_list=columns)
        elif stage == "validate":
            self.val_df = PandasDataset(casmi_val_df, column_list=columns)
        elif stage == "predict":
            # test.parquet carries no labels, so only the spectrum columns exist. molecule_id rides
            # along because scoring is per molecule, not per spectrum, and a molecule's spectra have
            # to be grouped back together afterwards.
            self.predict_df = PandasDataset(
                casmi_test_df,
                column_list=['molecule_id', 'precursor_mz', 'processed_mzs', 'processed_intensities'],
                shuffle=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_df,
            batch_size=self.batch_size,
            collate_fn=self.get_collator('fit'),
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_df,
            batch_size=self.batch_size,
            collate_fn=self.get_collator('validate'),
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_df,
            batch_size=self.predict_batch_size,
            collate_fn=self.get_collator('predict'),
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    def get_collator(self, stage):
        return partial(self._collator, stage=stage)

    def _collator(self, data, stage='fit'):
        mzs = [[row['precursor_mz']] + row['processed_mzs'].tolist() for row in data]  # prepend precursor to m/zs
        ints = [[PRECURSOR_INTENSITY] + row['processed_intensities'].tolist() for row in
                data]  # prepend 2. to intensities
        if stage == 'fit' or stage == 'validate':
            labels = [row['bpe_tokenized_smiles'] for row in data]

        precursor_mz = torch.tensor(
            [row["precursor_mz"] for row in data],
            dtype=torch.float32
        )

        collision_energy = torch.tensor(
            [
                row.get("collision_energy_ev", 0.0)[0]
                if isinstance(row.get("collision_energy_ev", 0.0), (list, np.ndarray))
                else row.get("collision_energy_ev", 0.0)
                for row in data
            ],
            dtype=torch.float32,
        )

        adduct = torch.tensor(
            [ADDUCT_MAP.get(row.get("adduct"), -1) for row in data]
        )
        ion_mode = torch.tensor(
            [
                ION_MODE_MAP.get(
                    row.get("ionization_mode", "unknown"),
                    -1
                )
                for row in data
            ]
        )
        # collate peaks
        max_peak_len = max([len(x) for x in mzs])
        mzs = [list(x) + [MZ_PAD_ID] * (max_peak_len - len(x)) for x in mzs]
        ints = [list(x) + [INTENSITY_PAD_ID] * (max_peak_len - len(x)) for x in ints]
        mz_array = torch.tensor(mzs)
        intensity_array = torch.tensor(ints)
        attention_mask = torch.where(mz_array == MZ_PAD_ID, 0,
                                     1)  # attention_mask is being created to ignore padding tokens

        metadata = torch.tensor(
            list(
                zip(
                    precursor_mz,
                    collision_energy,
                    adduct,
                    ion_mode,
                )
            ),
            dtype=torch.float32,
        )

        # collate labels
        batch = {}
        if stage == 'fit' or stage == 'validate':
            max_smiles_len = max([len(x) for x in labels])
            labels = [list(x) + [BPE_PAD_ID] * (max_smiles_len - len(x)) for x in labels]
            label_array = torch.tensor(labels)

            batch['mzs'] = mz_array
            batch['intensities'] = intensity_array
            batch['attention_mask'] = attention_mask
            batch['structure_tokens'] = label_array  # tokenized labels for training
            batch['metadata'] = metadata
        if stage == 'validate':
            batch['smiles'] = [row['normalized_smiles'] for row in
                               data]  # include original smiles for validation metrics
        else:
            batch = batch | {
                'mzs': mz_array,
                'intensities': intensity_array,
                'attention_mask': attention_mask,
                'metadata': metadata
            }
        if stage == 'predict':
            batch['molecule_id'] = [row['molecule_id'] for row in data]
        return batch


