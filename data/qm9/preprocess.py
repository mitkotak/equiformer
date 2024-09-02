#""Exposes the QM9 datasets in a convenient format."""

from typing import Dict, Iterator

from typing import Dict, Iterator
from sklearn.preprocessing import StandardScaler
from absl import logging
import torch_geometric as pyG
import numpy as np
import torch

def add_edges_transform(radial_cutoff: float, add_self_edges: bool):
    """Returns a PyG transform that adds edges to the graph."""

    def add_edges(data):
        data.edge_index = pyG.nn.radius_graph(
            data.pos, r=radial_cutoff, loop=add_self_edges
        ).numpy()
        return data

    return add_edges


def split_dataset(
    dataset: torch.utils.data.Dataset, splits: Dict[str, int], seed: int
) -> Dict[str, torch.utils.data.Dataset]:
    if splits.get("test") is None:
        splits["test"] = len(dataset) - splits["train"] - splits["val"]
    if splits["test"] < 0:
        raise ValueError(
            f"Number of test graphs ({splits['test']}) cannot be negative."
        )
    datasets = torch.utils.data.random_split(
        dataset,
        [splits["train"], splits["val"], splits["test"]],
        generator=torch.Generator().manual_seed(seed),
    )
    return {
        "train": datasets[0],
        "val": datasets[1],
        "test": datasets[2],
    }


def _nearest_multiple_of_8(x: int) -> int:
    return int(np.ceil(x / 8) * 8)


def estimate_padding(
    graph: pyG.data.Data, cutoff: float, add_self_edges: bool, batch_size: int
) -> Dict[str, int]:
    """Estimates the padding needed to batch the graphs."""
    n_node = int(graph.pos.shape[0])
    n_edge = pyG.nn.radius_graph(graph.pos, r=cutoff, loop=add_self_edges).shape[1]
    return dict(
        n_node=_nearest_multiple_of_8(n_node * batch_size),
        n_edge=_nearest_multiple_of_8(n_edge * batch_size),
        n_graph=batch_size,
    )

def pad_graph(data: pyG.data.Data, n_node: int, n_edge: int) -> pyG.data.Data:
    """Pads a PyG Data object with dummy nodes and edges."""
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    
    # Padding nodes
    if num_nodes < n_node:
        pad_nodes = torch.zeros((n_node - num_nodes, data.x.size(1)))
        data.x = torch.cat([data.x, pad_nodes], dim=0)
    
    # Padding edges
    if num_edges < n_edge:
        pad_edges = torch.zeros((2, n_edge - num_edges), dtype=torch.long)
        pad_edge_attr = torch.zeros((n_edge - num_edges, data.edge_attr.size(1)))
        data.edge_index = torch.cat([data.edge_index, pad_edges], dim=1)
        data.edge_attr = torch.cat([data.edge_attr, pad_edge_attr], dim=0)
    
    return data

def to_graph(data: pyG.data.Data, target_property_index: int) -> pyG.data.Data:
    data = pyG.data.Data(
        pos = data.pos,
        z = data.z,
        edge_index = data.edge_index,
        y = data.y[:, target_property_index],
        num_nodes = data.num_nodes)
    return data


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

        # Define the properties.
        properties = {
            0: {"name": "mu", "description": "Dipole moment", "unit": "D"},
            1: {
                "name": "alpha",
                "description": "Isotropic polarizability",
                "unit": "a_0^3",
            },
            2: {
                "name": "epsilon_HOMO",
                "description": "Highest occupied molecular orbital energy",
                "unit": "eV",
            },
            3: {
                "name": "epsilon_LUMO",
                "description": "Lowest unoccupied molecular orbital energy",
                "unit": "eV",
            },
            4: {
                "name": "Delta epsilon",
                "description": "Gap between epsilon_HOMO and epsilon_LUMO",
                "unit": "eV",
            },
            5: {
                "name": "R^2",
                "description": "Electronic spatial extent",
                "unit": "a_0^2",
            },
            6: {
                "name": "ZPVE",
                "description": "Zero point vibrational energy",
                "unit": "eV",
            },
            7: {"name": "U_0", "description": "Internal energy at 0K", "unit": "eV"},
            8: {"name": "U", "description": "Internal energy at 298.15K", "unit": "eV"},
            9: {"name": "H", "description": "Enthalpy at 298.15K", "unit": "eV"},
            10: {"name": "G", "description": "Free energy at 298.15K", "unit": "eV"},
            11: {
                "name": "c_v",
                "description": "Heat capavity at 298.15K",
                "unit": "cal/(mol K)",
            },
            12: {
                "name": "U_0_ATOM",
                "description": "Atomization energy at 0K",
                "unit": "eV",
            },
            13: {
                "name": "U_ATOM",
                "description": "Atomization energy at 298.15K",
                "unit": "eV",
            },
            14: {
                "name": "H_ATOM",
                "description": "Atomization enthalpy at 298.15K",
                "unit": "eV",
            },
            15: {
                "name": "G_ATOM",
                "description": "Atomization free energy at 298.15K",
                "unit": "eV",
            },
            16: {"name": "A", "description": "Rotational constant", "unit": "GHz"},
            17: {"name": "B", "description": "Rotational constant", "unit": "GHz"},
            18: {"name": "C", "description": "Rotational constant", "unit": "GHz"},
        }
        # Search for the target property.
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

        # Split the dataset.
        dataset = pyG.datasets.QM9(
            root=root,
            transform=add_edges_transform(
                radial_cutoff=radial_cutoff, add_self_edges=add_self_edges
            ),
        )
        self.datasets = split_dataset(dataset, splits, seed)

    def to_pyg_graphs(
        self, split: str, batch_size: int
    ) -> Iterator[jraph.GraphsTuple]:
        """Returns batched and padded graphs."""
        logging.info(f"Creating {split} dataset.")

        batch = []
        padding = None

        while True:
            for data in self.datasets[split]:
                data_as_pyg = to_graph(
                    data, target_property_index=self.target_property_index
                )
                batch.append(data_as_pyg)

                if len(batch) == batch_size:
                    batch_pyG = pyG.data.Batch.from_data_list(batch)

                    # Compute padding if not already computed.
                    if padding is None:
                        padding = dict(
                            n_node=int(
                                2.0 * np.ceil(batch_pyG.num_nodes.sum() / 64) * 64
                            ),
                            n_edge=int(
                                2.0 * np.ceil(batch_pyG.num_edges.sum() / 64) * 64
                            ),
                            n_graph=batch_size + 1,
                        )
                        logging.info(f"Split {split}: Padding computed as {padding}")

                    batch_pyG = pad_graph(batch_pyG, n_node=padding['n_node'], n_edge=padding['n_edge'])
                    yield batch_pyG

                    batch = []

    def get_datasets(self, batch_size: int) -> Dict[str, Iterator[jraph.GraphsTuple]]:
        """Returns the splits of the dataset."""
        return {
            split: self.to_pyg_graphs(split, batch_size)
            for split in self.datasets.keys()
        }