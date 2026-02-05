from dataclasses import dataclass, field


@dataclass
class RunConfig:
    photons_per_batch_forward: int = field(
        default=10_000,
        metadata={
            "unit": "",
            "description": "How many photons to process in each forward pass",
        },
    )
    photons_per_batch_backward: int = field(
        default=5_000,
        metadata={
            "unit": "",
            "description": "How many photons to process in each backward pass",
        },
    )
    batches_forward: int = field(
        default=10,
        metadata={"unit": "", "description": "Number of forward passes"},
    )
    batches_backward: int = field(
        default=5,
        metadata={"unit": "", "description": "Number of backward passes"},
    )
    processes: int = field(
        default=8,
        metadata={
            "unit": "",
            "description": "Number of processes to use in multiprocessing, ignored in linear execution",
        },
    )
