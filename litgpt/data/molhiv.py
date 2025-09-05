from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader

from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle
from litgpt.tokenizer import Tokenizer

from ogb.graphproppred import PygGraphPropPredDataset
import pandas as pd
import numpy as np
import json
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

@dataclass
class MolHIV(DataModule):
    mask_prompt: bool = False
    prompt_style: Union[str, PromptStyle] = "hiv-molecule-2-activity"
    ignore_index: int = -100
    seed: int = 42
    num_workers: int = 4
    include_multiturn_conversations: bool = False

    ogb_name: str = 'ogbg-molhiv'
    ogb_root: str = 'data'
    download_dir: Path = Path(f"./{ogb_root}/{ogb_name.replace("-", "_")}")

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    val_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        torch.serialization.add_safe_globals([DataEdgeAttr])
        torch.serialization.add_safe_globals([GlobalStorage])
        torch.serialization.add_safe_globals([DataTensorAttr])
        torch.serialization.safe_globals([DataTensorAttr])

        pyg_data = PygGraphPropPredDataset(name=self.ogb_name, root=self.ogb_root)

        smiles = pd.read_csv(f'{self.download_dir}/mapping/mol.csv.gz')['smiles'].tolist()
        activities = [
            "active" if pyg_d.y.item() == 1 else "inactive"
            for pyg_d in pyg_data
        ]

        list_data = [
            {
                "instruction": "Classify the provided HIV molecule into \"active\" or \"inactive\"",
                "input": smile,
                "output": activity
            }
            for smile, activity in zip(smiles, activities)
        ]
        with open(f"{self.download_dir}/data.json", "w") as f:
            json.dump(list_data, f, indent=2, ensure_ascii=False)

    def setup(self, stage: str = "") -> None:
        with open(f"{self.download_dir}/data.json", encoding="utf-8") as f:
            data = json.load(f)

        split_dirt = f"{self.download_dir}/split/scaffold"
        train_idx = pd.read_csv(f"{split_dirt}/train.csv.gz").squeeze("columns").values
        valid_idx = pd.read_csv(f"{split_dirt}/valid.csv.gz").squeeze("columns").values
        # test_idx = pd.read_csv(f"{split_dirt}/test.csv.gz").squeeze("columns").values

        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in valid_idx]

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

        self.val_dataset = SFTDataset(
            data=val_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(max_seq_length=self.max_seq_length, ignore_index=self.ignore_index),
        )
