"""Collection of build methods for the different types of datasets.

To start, this will only include ports of the CUDA-based dimensionless simulations from the original paper. However, other methods such as file-based storage might be implemented."""

from numba import cuda as ncuda

from imfcs_pytorch.data.simulation.interface import (
    SimulationInterface,
    SimulationInterfaceLegacyModSig,
)
from imfcs_pytorch.data.simulation.simulator.sim_2d_dimensionless import (
    simulate_scan_func as sim_2d_dimensionless,
)
from imfcs_pytorch.data.simulation.simulator.sim_3d_dimensionless import (
    simulate_scan_func as sim_3d_dimensionless,
)
from imfcs_pytorch.data.simulation.simulator.sim_2d import simulate as sim_2d_physical
from imfcs_pytorch.data.simulation.simulator.sim_3d import simulate as sim_3d_physical
from imfcs_pytorch.data.simulation.utils.numba_funcs import cuda_compile


# Typing-specific imports
from yacs.config import CfgNode


# TODO: Perhaps a more elegant way to handle simulation start/stop?
def build_simulator(cfg: CfgNode, device: str) -> SimulationInterface:
    """Builder function for a simulator/queue class.

    This is necessary as simulators happen on a separate thread, and we start/teardown these threads using dedicated `start_simulation()` and `stop_simulation()` calls.

    If we obfuscate the simulator build step, we lose access to the `stop_simulation()` call, which means we will be stuck with zombie threads that cannot be terminated elegantly.

    For now, this is a workaround for that, though this might change when we start introducing datasets that work with disk-reads (such as pre-generated datasets).

    Args:
        cfg (CfgNode): YACS config object holding the transform setting parameters.
        device (str): Device to use for simulations.

    Returns:
        SimulationInterface: Simulator object.
    """
    # Setting the Numba CUDA context to a specific CUDA device ID.
    # This was added to ensure that simulation and PyTorch training occurs on the same device.
    cuda_device_ind = int("".join(filter(str.isdigit, device)))
    ncuda.select_device(cuda_device_ind)

    # Define the simulation function
    if cfg.SIMULATION.TYPE == "SIM_2D_1P_DIMLESS":
        simulate_scan_func = sim_2d_dimensionless
    elif cfg.SIMULATION.TYPE == "SIM_3D_1P_DIMLESS":
        simulate_scan_func = sim_3d_dimensionless
    elif cfg.SIMULATION.TYPE == "SIM_2D_PHYSICAL":
        simulate_scan_func = sim_2d_physical
    elif cfg.SIMULATION.TYPE == "SIM_3D_PHYSICAL":
        simulate_scan_func = sim_3d_physical
    else:
        raise ValueError(
            f"Invalid SIMULATION.TYPE {cfg.SIMULATION.TYPE} for build_simulator. Perhaps you forgot to add it to the `build_simulator` function (imfcs_pytorch/builders/dataset.py)?"
        )

    # Determine the simulator interface, i.e. 'modsig' (legacy) or 'minmax'
    # We define the values here to softly enforce that any new interfaces need to have the same input args.
    if cfg.SIMULATION.PARAMETER_SAMPLING_STRATEGY == "minmax":
        simulator_interface = SimulationInterface
    elif cfg.SIMULATION.PARAMETER_SAMPLING_STRATEGY == "modsig":
        simulator_interface = SimulationInterfaceLegacyModSig

    # Determine the nature of the simulation, i.e. 2D or 3D.
    # This is necessary since we need to pre-define the particle positions array.
    # In the 2D case, this will have 2 position vectors, x and y
    # In the 3D case, this will have 3: x, y and z
    # In the physical cases, there are 2 additional dimensions to track the bleaching and triplet-blink spaces.
    # This will enforce a rule that any new added simulations need to have a marker indicating it's dimensionlity.
    # Currently, the decision is taken to require a DIMENSIONS key under the constants defined for each simulation.
    simulator = simulator_interface(
        simulation_type=cfg.SIMULATION.TYPE,
        total_simulations=cfg.TRAINING.ITERATIONS * cfg.DATALOADER.PER_STEP_BATCH_SIZE,
        stacks_per_simulation=cfg.SIMULATION.BACKEND.SIM_COUNT,
        target_model_params=cfg.TASK[cfg.EXPERIMENT.TASK]["TARGETS"],
        simulator_function=cuda_compile(
            max=cfg.SIMULATION.BACKEND.SIM_COUNT,
            seed=cfg.EXPERIMENT.SEED,
            batch_size=cfg.DATALOADER.PER_STEP_BATCH_SIZE,
            device_id=cuda_device_ind,
        )(simulate_scan_func),
        sim_params_universal_constants=cfg.SIMULATION.CONSTANTS,
        sim_params_model_specific_constants=cfg.SIMULATION[cfg.SIMULATION.TYPE][
            "CONSTANTS"
        ],
        sim_params_model_specific_variables=cfg.SIMULATION[cfg.SIMULATION.TYPE][
            "VARIABLES"
        ],
        sim_dimensionality=cfg.SIMULATION[cfg.SIMULATION.TYPE]["DIMENSIONALITY"],
    )

    return simulator
