"""Exposes the QM9 datasets in a convenient format."""

from typing import Dict, Iterator

from sklearn.preprocessing import StandardScaler
from absl import logging
import torch_geometric as pyg
import torch
from torch_geometric.data import Data, Batch

from src.data import utils


def to_data(data: pyg.data.Data, target_property_index: int) -> Data:
    """Converts a datum from a PyG dataset to a PyG Data object."""
    # Compute relative positions
    pos = data.pos
    edge_index = data.edge_index
    relative_vectors = pos[edge_index[1]] - pos[edge_index[0]]

    return Data(
        pos=data.pos,
        z=data.z,
        y=data.y[:, target_property_index],
        edge_index=edge_index,
        edge_attr=relative_vectors,
    )


class QM9Dataset(torch.utils.data.Dataset):
    """Exposes the QM9 dataset in a convenient format."""

    def __init__(
        self,
        root: str,
        target_property: str,
        radial_cutoff: float,
        add_self_edges: bool,
        splits: Dict[str, int],
        seed: int,
    ):
        self.radial_cutoff = radial_cutoff
        self.add_self_edges = add_self_edges

        # Define the properties
        properties = {
            0: {"name": "mu", "description": "Dipole moment", "unit": "D"},
            1: {
                "name": "alpha",
                "description": "Isotropic polarizability",
                "unit": "a_0^3",
            },
            # ... (rest of the properties remain the same)
        }

        # Search for the target property
        target_property_index = None
        for key, value in properties.items():
            if value["name"] == target_property:
                target_property_index = key
                logging.info(
                    f"Target property {target_property}: {value['description']} ({value['unit']})"
                )
                break
        if target_property_index is None:
            raise ValueError(
                f"Unknown target property {target_property}. Available properties are: {', '.join([value['name'] for value in properties.values()])}"
            )
        self.target_property_index = target_property_index

        # Split the dataset
        dataset = pyg.datasets.QM9(
            root=root,
            transform=utils.add_edges_transform(
                radial_cutoff=radial_cutoff, add_self_edges=add_self_edges
            ),
        )
        self.datasets = utils.split_dataset(dataset, splits, seed)

    def to_pyg_graphs(
        self, split: str, batch_size: int
    ) -> Iterator[Batch]:
        """Returns batched graphs."""
        logging.info(f"Creating {split} dataset.")

        dataloader = pyg.loader.DataLoader(
            self.datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
        )

        for batch in dataloader:
            yield Batch.from_data_list([
                to_data(data, target_property_index=self.target_property_index)
                for data in batch.to_data_list()
            ])

    def get_datasets(self, batch_size: int) -> Dict[str, Iterator[Batch]]:
        """Returns the splits of the dataset."""
        return {
            split: self.to_pyg_graphs(split, batch_size)
            for split in self.datasets.keys()
        }