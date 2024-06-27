"""This is a port of the original simulation code utilizing numba (https://github.com/ImagingFCS/ImFCS_FCSNet_ImFCSNet/tree/master/imfcsnet/simulator)

This file provides an interface that facilitates integration into PyTorch training pipelines.

The idea here is to instantiate a single object which encapsulates generic `numba.cuda` simulation logic, which can be extended to different simulation types without needing bespoke code for every newly introduced simulation type.

By that rule, new simulations should be implemented within the framework provided, rather than breaking apart from this interface. These rules should be followed:
1. Simulations parameters should be defined within the context of the YACS configuration file (or, at the very least, as dictionaries). - This leverages Python dictionaries' property of ordering being maintained https://mail.python.org/pipermail/python-dev/2017-December/151283.html, which allows us to extract params by their indices, which is a necessary evil since Numba's CUDA implementations do not support dictionaries.*1
2. Simulation parameters should be numeric (ints or floats). Strings should be avoided at all costs. If a boolean variable is required (i.e. for flags, like DO_BLEACHING), use 0/1 to represent False/True.
3. Parameter sampling should be based on min/max sampling through a uniform distribution, log or natural.*2

To add new simulations, see additional documentation TODO: This needs to be added as a separate page.

*1 - True, dictionary ordering is only in Python 3.7 and up, but Python 3.6 was End-of-life'd in 2021... you really should update.
*2 - The original paper's "mod/sig" approach is retained for reproducibility experiments, but it is not recommended as it is slower, less interpretable, and just downright unnecessary when min/max just works fine (and is also what's reported in the paper).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import threading
import warnings
from numba import float32, uint64, int64

from imfcs_pytorch.data.simulation.utils.queue import SimulationQueue
from imfcs_pytorch.utils.config import extract_key_index_from_dict

# Typing-specific imports
from typing import Tuple, Union, List, Dict
from torchvision.transforms import v2 as T


class SimulationQueueDataset(Dataset):
    def __init__(
        self,
        total_sequences: int,
        simulation_queue: SimulationQueue,
        transforms: torch.nn.Module,
    ):
        """Wrapper around SimulationQueue objects to produce a PyTorch MapDataset.

        SimulationQueueDatasets should not be created manually. Instead, they should be instantiated within a SimulationInterface, which handles the parameter linking, thread spawning and teardown.

        Args:
            total_sequences (int): Total number of sequences to be simulated. Used for the __len__ property for ending the iteration process.
            simulation_queue (SimulationQueue): SimulationQueue object. Generally handled within the SimulationInterface.
            transforms (torch.nn.Module): PyTorch transformations. Generally composed using torchvision.transforms.v2.Compose.
        """
        self.total_sequences = total_sequences
        self.simulation_queue = simulation_queue
        self.transforms = transforms

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        # Simulation queue might be blocked. Keep a retry loop.
        while True:
            try:
                batch = (
                    self.simulation_queue.get_result_from_queue()
                )  # Get batch from queue
                # Unpack batch into inputs and targets.
                inputs, targets = batch

                # Apply transformations to input if needed.
                if self.transforms:
                    inputs = self.transforms(inputs)

                return (
                    inputs,
                    targets,
                )  # Yield the batch in the form of a tuple (input, target)
            except IndexError:
                print("Queue empty, waiting for retry.")


class SimulationInterface:
    def __init__(
        self,
        simulation_type: str,
        total_simulations: int,
        stacks_per_simulation: int,
        target_model_params: List[str],
        simulator_function: callable,
        sim_params_universal_constants: Dict[str, Union[int, float]],
        sim_params_model_specific_constants: Dict[str, Union[int, float]],
        sim_params_model_specific_variables: Dict[
            str, Dict[str, Union[int, float, str]]
        ],
        sim_dimensionality: int = 4,
        max_queue_size: int = 120000,
        max_queue_retry_time: int = 3,
    ):
        """Interface that instantiates the simulation queue, and provides access to a linked PyTorch Dataset object.

        Handles the management of the simulation queue, random value sampling and thread teardown. These are parts which make writing CUDA kernels a bit difficult, but these are abstracted away in this interface, provided the expected configuration semantics are followed.

        While messy, this implementation of the simulator interface has the following benefits:
        1. Fast. Since simulations happen on a separate thread and write to a queue, simulations do not block the model training thread.
        2. Extensible. The simulation interface might take some getting used to, but it allows different simulations to be implemented within the framework without needing bespoke code.

        Args:
            simulation_type (str): The simulation identifier string. Generally defined under `SIMULATION.TYPE` in the YACS config object. Used during the pre-allocation of the particle arrays.
            total_simulations (int): The total number of simulations to run. This will attempt to stop the execution of the simulation thread.
            stacks_per_simulation (int): The number of image stacks to simulate on the CUDA kernel execution. 4096 as per the original paper was tested to work with <1GB of VRAM use.
            simulator_function (callable): The simulator function to execute as a CUDA kernel. Generally a compiled function via `imfcs_pytorch.data.simulation.utils.numba_funcs.cuda_compile`.
            sim_params_universal_constants (Dict[str, Union[int, float]]): The universal constants to use for the simulation (number of pixels, frames, etc). Generally accessed via SIMULATION.CONSTANTS from the YACS config object.
            sim_params_model_specific_constants (Dict[str, Union[int, float]]): Constants specific to the type of simulation (magnification, numerical aperture etc.). Generally accessed via the SIMULATION.<SIMULATION.TYPE>.CONSTANTS key in the YACS config object.
            sim_params_model_specific_variables (Dict[str, Dict[str, Union[int, float, str]]]): Variable parameters specific to the type of simulation (D1, NO_OF_PARTICLES etc.). Should consists of a `MIN`, `MAX` and `TRANSFORM` key to define the sampling methods. See the `sample_simulation_variables` method for details on the sampling process. Generally accessed via the SIMULATION.<SIMULATION.TYPE>.VARIABLES key in the YACS config object.
            sim_dimensionality (int, optional): Dimensionlity of the simulation. Used to define the particle state array. For example, for SIM_2D_PHYSICAL, the array has 4 dimensions: x_pos, y_pos, bleach_state and blink_state. Generally accessed via the SIMULATION.<SIMULATION.TYPE>.DIMENSIONALITY key in the YACS config object. Defaults to 4.
            max_queue_size (int, optional): Maximum queue size to use to store simulations. Simulations are written to a queue, which is popped via the PyTorch Dataset. `max_queue_size=0` means an infinite queue size, which is not recommend due to ever-increasing memory use. Defaults to 120000.
            max_queue_retry_time (int, optional): The time to wait before retrying during queue operations (put and pop operations) in seconds. For example, if the queue is full and `max_queue_retry_time=3`, the simulator will wait 3 seconds before attempting another put operation. Defaults to 3.
        """
        # Storing simulation type
        # This is technically not important, but we need it to dynamically pre-allocate the particle positions array.
        self.simulation_type = simulation_type

        # Storing initialization variables.
        self.stacks_per_simulation = stacks_per_simulation
        self.target_model_params = target_model_params
        self.simulator_function = simulator_function

        # Initializing necessary arrays.
        self.total_sequences = total_simulations

        # Simulation parameters.
        # # Constants: These variables are those that remain constant throughout the simulation process.
        self.sim_pixels = sim_params_universal_constants["NUM_PIXELS"]
        self.sim_frames = int64(sim_params_universal_constants["FRAMES"])
        self.sim_margin = int64(sim_params_universal_constants["MARGIN"])
        self.sim_width = float32(self.sim_pixels + 2 * self.sim_margin)
        self.sim_dimensionality = sim_dimensionality

        # This mapping is used to convert the string representations of transformations into an integer form, which can be used to call mask-vectorized transformation functions during the sampling process.
        self.transform_string_int_mapping = {
            None: 0,
            "log": 1,
        }

        # Validate and create the sampling parameter arrays.
        # Note that these are not CUDA-kernel ready, as they need to be extended to match the self.total_sequences value previously designated.
        self.validate_sim_params_const(param_dict=sim_params_universal_constants)
        self.validate_sim_params_const(param_dict=sim_params_model_specific_constants)
        self.validate_sim_params_var(
            param_min_dict=sim_params_model_specific_variables["MIN"],
            param_max_dict=sim_params_model_specific_variables["MAX"],
            param_transform_dict=sim_params_model_specific_variables["TRANSFORM"],
            transform_string_int_mapping=self.transform_string_int_mapping,
        )

        # Store a reference to the sim_params_model_specific_variables array.
        # This is used during the initialization of the simulation queue system
        self.sim_params_model_specific_variables = sim_params_model_specific_variables

        # Build array of constant simulation parameters. This array should remain constant throughout the simulations.
        self.sim_params_arr_const = self.build_sim_params_arr_const(
            param_dict=sim_params_universal_constants
        )
        self.model_specific_params_arr_const = self.build_sim_params_arr_const(
            param_dict=sim_params_model_specific_constants
        )
        # Build array for sampling of variable simulation parameters. This is used for vectorized sampling through the shape, which contains the [mean, sigma, clip_min, clip_max, transform].
        self.sim_params_arr_var = self.build_sim_params_arr_var(
            param_min_dict=sim_params_model_specific_variables["MIN"],
            param_max_dict=sim_params_model_specific_variables["MAX"],
            param_transform_dict=sim_params_model_specific_variables["TRANSFORM"],
            transform_string_int_mapping=self.transform_string_int_mapping,
        )

        # Since the constant parameters do not change across iterations, we can initialize the constants array during the __init__ stage, and then reuse it across every simulation iteration.
        # We use `np.broadcast_to` to duplicate this array across the total sequences we are simulating.
        self.sim_params_constants = np.stack(
            [self.sim_params_arr_const] * (self.stacks_per_simulation),
            0,
        )
        self.model_specific_constants = np.stack(
            [self.model_specific_params_arr_const] * (self.stacks_per_simulation),
            0,
        )

        # Creating the simulation queue.
        # This allows us to simulate without blocking the main execution thread.
        # We need to create the simulation parameter sampling function, which is required to produce the function signature of the simulation_function.
        # We also need to pre-extract the indices of the target. This will be used to prepare the network targets corresponding to the input batch.
        # Step 1: Find the index of the target variables based on the provided input variable - self.target_model_params
        self.target_indices = []
        for target_param_name in self.target_model_params:
            self.target_indices.append(
                extract_key_index_from_dict(
                    sim_params_model_specific_variables["MIN"], target_param_name
                )
            )

        # Initialize and start the simulation queue.
        self.simulation_queue = SimulationQueue(
            simulation_function=self.simulator_function,
            simulation_parameter_function=self.parameter_generation,
            total_simulations=self.total_sequences,
            max_queue_size=max_queue_size,
            retry_time=max_queue_retry_time,
        )

        # Do not start simulating here.
        self.simulations_started = False

    def parameter_generation(
        self,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Helper function with no input arguments which can generate the parameters to be used in the queue.

        Returns:
            Dict[str, np.ndarray]: Parameter dictionary to be passed into the queue.
            np.ndarray: Target array. This is used to add both the x (input) and y (target) to the queue simultaneously as a tuple. This allows the queue to follow PyTorch's Dataset semantics.
        """
        _max_number_of_particles = self.get_max_number_of_particles(
            self.simulation_type,
            sim_params_model_specific_variables=self.sim_params_model_specific_variables,
        )

        # Sample the parameters here.
        sampled_parameters = self.sample_simulation_variables(
            sim_params_arr_var=self.sim_params_arr_var,
            total_sequences=self.stacks_per_simulation,
        )

        # Extract the target values based on the specified indices.
        target_arr = sampled_parameters[:, self.target_indices]

        scan_arr = np.zeros(
            (
                self.stacks_per_simulation,
                self.sim_frames,
                self.sim_pixels,
                self.sim_pixels,
            ),
            dtype=np.int32,
        )

        # TODO: I hate this code since it breaks modularity, but it's a tentative measure to stop a blocker of needing to initialize the positions array in CPU, as CUDA cannot assign new arrays. RETURN HERE TO THINK OF A MORE ELEGANT WAY!
        # Particles position array also depends on the simulation type too, i.e. 2 position axes for 2D, 3 position axes for 3D etc.
        # Seems like yet another thing that needs to be split out.
        param_dict = {
            "idx_start": 0,
            "numSeq": self.stacks_per_simulation,
            # "seq_id": ,   # seq_id should be defined by the thread number. Makes this a needed variable in the queue code.
            "scan": scan_arr.astype(
                np.int32
            ),  # scan needs to be a unique array each time, since the results need to be written into the queue each time. Remember: CUDA cannot return values, only write to an array.
            "model_param_store": sampled_parameters.astype(
                np.float32
            ),  # These are the variable parameters, which need to be generated each time.
            "sim_param_store": self.sim_params_constants.astype(np.float32),
            "model_specific_param_store": self.model_specific_constants.astype(
                np.float32
            ),
            "position": self.init_particle_state_array(
                self.stacks_per_simulation,
                _max_number_of_particles,
                self.sim_dimensionality,
            ).astype(np.float32),
            # "emccd_cdf": self.emccd_cdf.astype(np.float32),
            # "numFrames": ,
        }
        # Return 2 values in the form of kwargs_dict, network_target
        return param_dict, target_arr

    def start_simulation(self):
        """Start the simulation on a separate thread. Remember to terminate simulations upon conclusion of training by calling `stop_simulation()`.

        This allows simulations to happen simultaneously with the model training process with minimal blocking.
        """
        self.simulations_started = True

        # This starts the simulation queue on a separate non-blocking thread.
        # See: https://stackoverflow.com/a/27261365 for why we use a lambda function to signal the start and stop
        self.simulation_thread = threading.Thread(
            target=self.simulation_queue.simulation_worker,
            daemon=True,
            args=(lambda: self.simulations_started,),
        )
        self.simulation_thread.start()

    def stop_simulation(self):
        """Teardown simulation thread cleanly.

        Threads had to be used over multiprocessing (which provides a .terminate() option) as CUDA kernels cannot be pickled. This leads to a more difficult teardown process, but this provides a relatively proper teardown interface to prevent zombie threads.
        """
        # Send stop signal to threads.
        # Why use threads over multiprocessing with .terminate()?
        # The CUDA kernels cannot be pickled, so we're forced to use threads.

        # Set the flag to False to prevent creation of new datasets using this simulation queue.
        # Since this property was set as part of the lambda function, it should be able to force stop the infinite thread.
        # See: https://stackoverflow.com/a/27261365
        # This flag should call break in the infinite while loop.
        self.simulations_started = False

        # Now that simulations are stopped, we need to empty the queue.
        # This prevents a deadlock.
        self.simulation_queue.clear_queue()

        # Wait for simulation threads to complete
        self.simulation_thread.join()

    def get_max_number_of_particles(
        self,
        simulation_type: str,
        sim_params_model_specific_variables: dict,
    ) -> int:
        """Determine the maximum possible number of particles.

        In general, any simulation should have a "NO_OF_PARTICLES" key which allows us to define the maximum number of particles. This will allow us to pre-allocate the NumPy array which will be sent to the compiled CUDA simulation kernels (which are unable to dynamically allocate matrices on-demand).

        This function is necessary as the original dimensionless simulations reformulate number of particles as 'ParticleDensity', which needs to be reverse converted to obtain the number of particles.

        Args:
            simulation_type (str): Simulation identifier string. Generally extracted from SIMULATION.TYPE in the YACS config object.
            sim_params_model_specific_variables (dict): The simulation variables dictionary. Generally defined under SIMULATION.<SIM_NAME>.VARIABLES in the YACS config object. This function expects there to be a `MAX` key.

        Returns:
            int: The maximum possible number of particles for this simulation setup.
        """
        # Part of the ugly code to pre-allocate the positions array. We need to create the particle `positions` array on CPU since CUDA cannot generate them. Also, since the number of particles varies each simulation, we can't dynamically generate the positions array.
        # Instead, we pre-extrct the maximum possible number of particles as per the original code.
        if simulation_type in ["SIM_2D_1P_DIMLESS", "SIM_3D_1P_DIMLESS"]:
            _max_particle_density = sim_params_model_specific_variables["MAX"][
                "PARTICLE_DENSITY"
            ]
            _max_no_of_particles = uint64(
                _max_particle_density * self.sim_width * self.sim_width
            )

            return _max_no_of_particles
        elif simulation_type in [
            "SIM_2D_PHYSICAL_PARTICLE_DENSITY",
            "SIM_3D_PHYSICAL_PARTICLE_DENSITY",
        ]:
            # These simulations have a modular margin parameter, so we need to calculate the theoretical maximum
            # This means the maximum of PARTICLE_DENSITY and minimum of MARGIN
            _max_particle_density = sim_params_model_specific_variables["MAX"][
                "PARTICLE_DENSITY"
            ]
            _min_margin = sim_params_model_specific_variables["MIN"]["MARGIN"]
            _max_no_of_particles = uint64(
                _max_particle_density * (self.sim_pixels + (2 * _min_margin)) ** 2
            )

            return _max_no_of_particles
        else:
            return sim_params_model_specific_variables["MAX"]["NO_OF_PARTICLES"]

    @staticmethod
    def init_particle_state_array(
        no_of_sequences: int, max_no_of_particles: int, dimensions: int = 4
    ) -> np.ndarray:
        """Helper function to generate a particle state array. This is necessary as CUDA kernels cannot allocate/initialize new arrays on-demand. Instead, we need to allocate these arrays on CPU then send them to device. This helper function creates the array.

        Note that the max number of particles needs to be determined before-hand, to save on an arithmetic operation being called every time this function call happens.

        Note that there might be some unused memory due to the initialization based on the max_no_of_particles. This is necessary since simulate multiple sequences simultaneously, and all share the same positions array. This means it's easier and safer to just assume all particles, as we can just ignore the unused particles in this array by looping.

        Args:
            no_of_sequences (int): Number of sequences (simultaneous simulations) to simulate in the compiled CUDA kernel.
            max_no_of_particles (int): The maximum number of particles possible with this simulation setup. Remember that CUDA kernels simultaneously perform multiple simulations which might have different particle counts, hence the need to pre-allocate the maximum possible array size.
            dimensions (int, optional): Dimensionlity of the simulation. Used to define the particle state array. For example, for SIM_2D_PHYSICAL, the array has 4 dimensions: x_pos, y_pos, bleach_state and blink_state. Generally accessed via the SIMULATION.<SIMULATION.TYPE>.DIMENSIONALITY key in the YACS config object. Defaults to 4.

        Returns:
            np.ndarray: The particle state array in the shape (no_of_sequences, max_no_of_particles, dimensions).
        """
        positions_arr = np.empty(
            (no_of_sequences, max_no_of_particles, dimensions), dtype=np.float32
        )
        return positions_arr

    @staticmethod
    def validate_sim_params_const(param_dict: Dict[str, Union[int, float]]):
        """Validation function for the constant simulation parameters.

        As the simulation parameters will be passed to the CUDA kernels as arrays, the values need to be numerics. This function performs a basic check to ensure that only int and float values exist.

        Args:
            param_dict (Dict[str, Union[int, float]]): Parameter dictionary of constants. Generally accessed via the SIMULATION.*.CONSTANTS keys in the YACS config object.

        Raises:
            ValueError: Invalid parameter type. Only int and float values are allowed for interfacing with the CUDA simulator.
        """
        # Populate the array with the values from the provided kwargs.
        for key, value in param_dict.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Expected int or float from param_dict. Got {{{key}: {value}}} of type {type(value)}."
                )

    @staticmethod
    def build_sim_params_arr_const(
        param_dict: Dict[str, Union[int, float]],
    ) -> np.ndarray:
        """Builder function that initializes the array of constant simulation parameters to be passed to the GPU execution thread.

        This should be used to produce the `simPar` array as used in the original codebase. Note that this produces a single array, and needs to be extended to cover total_sequences in order to enable parallelized execution as CUDA kernels.

        Ideally, this should only be initialized once as values do not change.

        Args:
            param_dict (Dict[str, Union[int, float]]): Parameter dictionary of constants. Generally accessed via the SIMULATION.*.CONSTANTS keys in the YACS config object.

        Returns:
            np.ndarray: Constant parameters as an array object. We can access the corresponding constant by name using the `imfcs_pytorch.utils.config.extract_key_index_from_dict` function, as Python dictionaries are ordered by insertion order.
        """
        # Note that we completely skip over the checking of seq_start and seq_end. This is because we are building an array from scratch, rather than assigning values into a pre-existing array. This is also the behaviour of the original codebase, where the arrays are built from (0, self.numSeq), which means the full sequence length is used.

        # Initializing the array based on the number of elements in the input kwargs array.
        sim_params_arr_const = np.empty(len(param_dict), dtype=np.float32)

        # Populate the array with the values from the provided kwargs.
        for ind, (key, value) in enumerate(param_dict.items()):
            # We should be able to directly insert the values, as validation was done in validate_sim_params_const()
            sim_params_arr_const[ind] = value

        return sim_params_arr_const

    @staticmethod
    def validate_sim_params_var(
        param_min_dict: Dict[str, float],
        param_max_dict: Dict[str, float],
        param_transform_dict: Dict[str, str],
        transform_string_int_mapping: Dict[int, str],
    ):
        """Validation function for the sampled simulation variables.

        Involves the checking of the min, max and uniform distribution type (natural/log).

        This should only be run once at the start of the program to ensure that all future sampling is not slowed by repeated validation steps.

        Args:
            param_min_dict (Dict[str, float]): Parameter dictionary of the simulation variable minimum sampling values in natural scale. Generally accessed via the SIMULATION.<SIMULATION.TYPE>.VARIABLES.MIN key in the YACS config object.
            param_max_dict (Dict[str, float]): Parameter dictionary of the simulation variable maximum sampling values in natural scale. Generally accessed via the SIMULATION.<SIMULATION.TYPE>.VARIABLES.MAX key in the YACS config object.
            param_transform_dict (Dict[str, str]): Parameter dictionary of the simulation variable uniform distribution transformation type. Generally accessed via the SIMULATION.<SIMULATION.TYPE>.VARIABLES.TRANSFORM key in the YACS config object.
            transform_string_int_mapping (Dict[int, str]): Valid uniform distribution transformations. Generally should include None: natural uniform, and "log": log-uniform.

        Raises:
            ValueError: Invalid parameters. MIN and MAX items should include only numerics (int and float), whereas the TRANSFORM items should be in [None, "log"].
        """
        # Step 1: Ensure that all arrays for the variable simulation parameters are of the same length.
        reference_length = len(
            param_max_dict
        )  # Use the max dictionary as the reference.
        for dict_name, d in zip(
            [
                "param_min_dict",
                "param_transform_dict",
            ],
            [
                param_min_dict,
                param_transform_dict,
            ],
        ):
            if len(d) != reference_length:
                raise ValueError(
                    f"Length of {dict_name} does not match length of param_max_dict. Got {len(d)}, expected {reference_length}."
                )

        # Step 2: Ensure that all keys match for all the parameter arrays.
        reference_keys = (
            param_max_dict.keys()
        )  # Extract the reference keys from the max dictionary.
        for dict_name, d in zip(
            [
                "param_min_dict",
                "param_max_dict",
            ],
            [
                param_min_dict,
                param_transform_dict,
            ],
        ):
            if set(d.keys()) != reference_keys:
                raise ValueError(
                    f"Keys in {dict_name} do not match the reference keys from param_mean_dict. Got {d.keys()}, expected {reference_keys}."
                )

        # Step 3: Ensure that the types of the arguments in each array are correct.
        # Check if values are floats for mean, std, min and max dictionaries.
        for dict_name, d in zip(
            [
                "param_min_dict",
                "param_max_dict",
            ],
            [
                param_min_dict,
                param_max_dict,
            ],
        ):
            for key, value in d.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Values in {dict_name} should be numeric floats or ints. Got {type(value)} for key {key}."
                    )

        # For the transform dictionary, check if the values are within the valid transformation types.
        for key, value in param_transform_dict.items():
            if value not in transform_string_int_mapping.keys():
                raise ValueError(
                    f"Valid options for values in param_transform_dict are {transform_string_int_mapping.keys()}. Got {value} for key {key}."
                )

    @staticmethod
    def build_sim_params_arr_var(
        param_min_dict: Dict[str, float],
        param_max_dict: Dict[str, float],
        param_transform_dict: Dict[str, str],
        transform_string_int_mapping: Dict[int, str],
    ) -> np.ndarray:
        """Builder function that initializes the variable sampling parameters, used in `sample_simulation_variables`.

        This builder only needs to be called once per-run, as it constructs the sampling parameters, not the sampled parameters.

        Args:
            param_min_dict (Dict[str, float]): Parameter dictionary of the simulation variable minimum sampling values in natural scale. Generally accessed via the SIMULATION.<SIMULATION.TYPE>.VARIABLES.MIN key in the YACS config object.
            param_max_dict (Dict[str, float]): Parameter dictionary of the simulation variable maximum sampling values in natural scale. Generally accessed via the SIMULATION.<SIMULATION.TYPE>.VARIABLES.MAX key in the YACS config object.
            param_transform_dict (Dict[str, str]): Parameter dictionary of the simulation variable uniform distribution transformation type. Generally accessed via the SIMULATION.<SIMULATION.TYPE>.VARIABLES.TRANSFORM key in the YACS config object.
            transform_string_int_mapping (Dict[int, str]): Valid uniform distribution transformations. Generally should include None: natural uniform, and "log": log-uniform.

        Returns:
            np.ndarray: Array that defines the sampling conditions for each variable in the shape (4, no_of_variables). The axes are [1: MIN, 2: MAX, 3: TRANSFORM, 4: IS_INT].
        """

        # Initializing the array based on the number of elements in the input kwargs array.
        # We already validated that the number of keys in each dictionary are identical using the validation function.
        # This means we can initialize a shape based on a reference, in this case, param_max.
        # This output array will have the shape (3, params_count)
        # The param-count-axis will be used for vectorized sampling through a uniform distribution (min [0], max [1]) by the sampling function.
        # The transformation axis will also be used to perform the required transformations.
        sim_params_arr_var = np.empty((4, len(param_max_dict)), dtype=np.float32)

        # Populate the array with the values from the provided kwargs.
        for ind, (min_val, max_val, transform_str) in enumerate(
            zip(
                param_min_dict.values(),
                param_max_dict.values(),
                param_transform_dict.values(),
            )
        ):
            # Determine of the value is an integer.
            # If either the min or max is not an integer, this will fail.
            # We don't explicitly check for types, as YACS verifies that type mismatches are not allowed.
            # Returns True if both min_val and max_val are integers.
            # False if not, i.e. a float.
            is_int = np.issubdtype(np.array([min_val, max_val]).dtype, np.integer)

            # We need to convert the transform index into the corresponding index value to store it in this array.
            # Based on [None, "log", "logit", "logitsample"]
            transform_index = transform_string_int_mapping[transform_str]
            sim_params_arr_var[:, ind] = np.array(
                [min_val, max_val, transform_index, is_int]
            )

        return sim_params_arr_var

    @staticmethod
    def sample_simulation_variables(
        sim_params_arr_var: np.ndarray, total_sequences: int
    ) -> np.ndarray:
        """A vectorized implementation of the sampling logic for simulation variables.

        Note: You might get NumPy warnings about invalid log() operations. That is normal, as vectorization means we perform some extraneous computation which applies log() to all variables, even those which might not use the log-uniform transform.

        Args:
            sim_params_arr_var (np.ndarray): The sampling parameters as built by `build_sim_params_arr_var`. Should be an array of the shape (4, no_of_variables), where the length-4 axis is ordered as [1: MIN, 2: MAX, 3: TRANSFORM, 4: IS_INT].
            total_sequences (int): The number of sequences to sample for. This number generally represents the number of parallel simulations to generate simultaneously.

        Returns:
            np.ndarray: The sampled simulation parameters, in the shape (total_sequences, params_count)
        """
        params_count = sim_params_arr_var.shape[1]

        par_min = sim_params_arr_var[0]
        par_max = sim_params_arr_var[1]
        par_transform = sim_params_arr_var[2]
        par_isint = sim_params_arr_var[3]

        # The minmax sampling method uses built-in NumPy functions to perfrom the sampling.
        # This increases readability overall.

        # For the purposes of log-uniform sampling, transform the min and max values based on the mask from par_transform
        par_min = np.where(
            par_transform == 1,  # When there is a "log": 1 transform
            np.log(par_min),
            par_min,  # No transform condition: 0
        )
        par_max = np.where(
            par_transform == 1,  # When there is a "log": 1 transform
            np.log(par_max),
            par_max,  # No transform condition: 0
        )

        rnd = np.random.uniform(
            low=par_min,
            high=par_max,
            size=(total_sequences, params_count),
        )

        # For the log-uniform samples, exponentiate them back to natural scale.
        rnd = np.where(
            par_transform == 1,
            np.exp(rnd),
            rnd,
        )

        # For values that were originally int values
        # Cast back to int by rounding.
        # We do not cast to integer, as the whole parameter array needs to remain as a specific type.
        # This is primarily for avoiding unrealistic target values
        # For example, NO_OF_PARTICLES = 1243.1871, CPS = 1241.141726
        rnd = np.where(par_isint == 1, np.round(rnd), rnd)

        return rnd

    def get_dataset(self, transforms: T = None) -> Dataset:
        """Create a PyTorch MapDataset that interfaces with the simulation queue.

        Args:
            transforms (torchvision.transforms.v2, optional): Transforms to apply to the image stacks during each __getitem__ call. Defaults to None.

        Raises:
            RuntimeError: Simulations have not been started yet. Call `start_simulation()` first before calling `get_dataset()`.

        Returns:
            Dataset: MapDataset which extracts simulations from the simulation queue.
        """
        # Check if the simulations have started. If not, raise an error.
        if not self.simulations_started:
            raise RuntimeError(
                "Simulation thread has not been started. Call `.start_simulation()` first before calling `.get_dataset()`"
            )

        return SimulationQueueDataset(
            self.total_sequences, self.simulation_queue, transforms
        )


# Legacy code will not be documented to the same standards. Refer to the canonical implementation above.
class SimulationInterfaceLegacyModSig(SimulationInterface):
    """Legacy support for the mod/sig parameter sampling method used by the original Tensorflow based code. Overrides the classes of the original."""

    def __init__(
        self,
        simulation_type: str,
        total_simulations: int,
        stacks_per_simulation: int,  # The total stacks to simulate per CUDA kernel call. 4096 tested to work with 1-3GB VRAM use.
        target_model_params: List[str],
        simulator_function: callable,
        sim_params_universal_constants: Dict[str, Union[int, float]],
        sim_params_model_specific_constants: Dict[str, Union[int, float]],
        sim_params_model_specific_variables: Dict[
            str, Dict[str, Union[int, float, str]]
        ],
        sim_dimensionality: int = 2,
        max_queue_size: int = 120000,  # 0 = Infinite queue size. Code tested for 4096 * 100, but not to the limit.
        max_queue_retry_time: int = 3,
    ):
        warnings.warn(
            "mod/sig-based parameter sampling is included for purposes of reproducing the original codebase, but is strongly discouraged since it introduces a layer of abstraction which makes intepretation much more difficult than it needs to be. min/max-based sampling is more robust and faster.",
            category=DeprecationWarning,
        )
        # This array will be defined based on the maximum possible number of particles.
        self.simulation_type = simulation_type

        # Storing initialization variables.
        self.stacks_per_simulation = stacks_per_simulation  # Controls the number of batches to simulate per kernel call.
        self.target_model_params = target_model_params
        self.simulator_function = simulator_function

        # Initializing necessary arrays.
        self.total_sequences = total_simulations

        # Simulation parameters.
        # # Constants: These variables are those that remain constant throughout the simulation process.
        self.sim_pixels = sim_params_universal_constants["NUM_PIXELS"]
        self.sim_frames = int64(sim_params_universal_constants["FRAMES"])
        self.sim_margin = int64(sim_params_universal_constants["MARGIN"])
        self.sim_width = float32(self.sim_pixels + 2 * self.sim_margin)
        self.sim_dimensionality = sim_dimensionality

        # This mapping is used to convert the string representations of transformations into an integer form, which can be used to call mask-vectorized transformation functions during the sampling process.
        self.transform_string_int_mapping = {
            None: 0,
            "log": 1,
            "logit": 2,
            "logitsample": 3,
        }

        # Validate and create the sampling parameter arrays.
        # Note that these are not CUDA-kernel ready, as they need to be extended to match the self.total_sequences value previously designated.
        self.validate_sim_params_const(param_dict=sim_params_universal_constants)
        self.validate_sim_params_const(param_dict=sim_params_model_specific_constants)
        self.validate_sim_params_var(
            param_mean_dict=sim_params_model_specific_variables["MEAN"],
            param_sigma_dict=sim_params_model_specific_variables["SIGMA"],
            param_min_dict=sim_params_model_specific_variables["MIN"],
            param_max_dict=sim_params_model_specific_variables["MAX"],
            param_transform_dict=sim_params_model_specific_variables["TRANSFORM"],
            transform_string_int_mapping=self.transform_string_int_mapping,
        )
        # Build array of constant simulation parameters. This array should remain constant throughout the simulations.
        self.sim_params_arr_const = self.build_sim_params_arr_const(
            param_dict=sim_params_universal_constants
        )
        self.model_specific_params_arr_const = self.build_sim_params_arr_const(
            param_dict=sim_params_model_specific_constants
        )
        # Build array for sampling of variable simulation parameters. This is used for vectorized sampling through the shape, which contains the [mean, sigma, clip_min, clip_max, transform].
        self.sim_params_arr_var = self.build_sim_params_arr_var(
            param_mean_dict=sim_params_model_specific_variables["MEAN"],
            param_sigma_dict=sim_params_model_specific_variables["SIGMA"],
            param_min_dict=sim_params_model_specific_variables["MIN"],
            param_max_dict=sim_params_model_specific_variables["MAX"],
            param_transform_dict=sim_params_model_specific_variables["TRANSFORM"],
            transform_string_int_mapping=self.transform_string_int_mapping,
        )

        # Since the constant parameters do not change across iterations, we can initialize the constants array during the __init__ stage, and then reuse it across every simulation iteration.
        # We use `np.broadcast_to` to duplicate this array across the total sequences we are simulating.
        self.sim_params_constants = np.stack(
            [self.sim_params_arr_const] * (self.stacks_per_simulation),
            0,
        )
        self.model_specific_constants = np.stack(
            [self.model_specific_params_arr_const] * (self.stacks_per_simulation),
            0,
        )

        # Creating the simulation queue.
        # This allows us to simulate without blocking the main execution thread.
        # We need to create the simulation parameter sampling function, which is required to produce the function signature of the simulation_function.

        # Part of the ugly code to pre-allocate the positions array. We need to create the particle `positions` array on CPU since CUDA cannot generate them. Also, since the number of particles varies each simulation, we can't dynamically generate the positions array.
        # Instead, we pre-extrct the maximum possible number of particles as per the original code.
        _max_particle_density = sim_params_model_specific_variables["MAX"][
            "PARTICLE_DENSITY"
        ]
        _max_no_of_particles = uint64(
            _max_particle_density * self.sim_width * self.sim_width
        )

        # We also need to pre-extract the indices of the target. This will be used to prepare the network targets corresponding to the input batch.
        # Step 1: Find the index of the target variables based on the provided input variable - self.target_model_params
        self.target_indices = []
        for target_param_name in self.target_model_params:
            self.target_indices.append(
                extract_key_index_from_dict(
                    sim_params_model_specific_variables["MIN"], target_param_name
                )
            )

        def parameter_generation() -> Tuple[Dict[str, np.ndarray], np.ndarray]:
            """Helper function with no input arguments which can generate the parameters to be used in the queue. Also

            Returns:
                Dict[str, np.ndarray]: Parameter dictionary to be passed into the queue.
                np.ndarray: Reference to the "scan" object in the dictionary. An ugly workaround for CUDA kernels' lack of ability to return values.
            """
            # Sample the parameters here.
            sampled_parameters = self.sample_simulation_variables(
                sim_params_arr_var=self.sim_params_arr_var,
                total_sequences=self.stacks_per_simulation,
            )

            # Extract the target values based on the specified indices.
            target_arr = sampled_parameters[:, self.target_indices]

            scan_arr = np.zeros(
                (
                    self.stacks_per_simulation,
                    self.sim_frames,
                    self.sim_pixels,
                    self.sim_pixels,
                ),
                dtype=np.int32,
            )

            # Particles position array also depends on the simulation type too, i.e. 2 position axes for 2D, 3 position axes for 3D etc.
            # Seems like yet another thing that needs to be split out.
            param_dict = {
                "idx_start": 0,
                "numSeq": self.stacks_per_simulation,
                # "seq_id": ,   # seq_id should be defined by the thread number. Makes this a needed variable in the queue code.
                "scan": scan_arr.astype(
                    np.int32
                ),  # scan needs to be a unique array each time, since the results need to be written into the queue each time. Remember: CUDA cannot return values, only write to an array.
                "model_param_store": sampled_parameters.astype(
                    np.float32
                ),  # These are the variable parameters, which need to be generated each time.
                "sim_param_store": self.sim_params_constants.astype(np.float32),
                "model_specific_param_store": self.model_specific_constants.astype(
                    np.float32
                ),
                "position": self.init_particle_state_array(
                    self.stacks_per_simulation,
                    _max_no_of_particles,
                    self.sim_dimensionality,
                ).astype(np.float32),
                # "emccd_cdf": self.emccd_cdf.astype(np.float32),
                # "numFrames": ,
            }
            # Return 2 values in the form of kwargs_dict, network_target
            return param_dict, target_arr

        # Initialize and start the simulation queue.
        self.simulation_queue = SimulationQueue(
            simulation_function=self.simulator_function,
            simulation_parameter_function=parameter_generation,
            total_simulations=self.total_sequences,
            max_queue_size=max_queue_size,
            retry_time=max_queue_retry_time,
        )

        # Do not start simulating here.
        self.simulations_started = False

    @staticmethod
    def validate_sim_params_var(
        param_mean_dict: Dict[str, float],
        param_sigma_dict: Dict[str, float],
        param_min_dict: Dict[str, float],
        param_max_dict: Dict[str, float],
        param_transform_dict: Dict[str, str],
        transform_string_int_mapping: Dict[int, str],
    ):
        """Validation function for the sampled simulation variables. Involves the checking of the mean/std, transformation type, and clipping values. This should only be run once at the start of the program to ensure that all future sampling is not slowed by repeated validation steps."""
        # Step 1: Ensure that all arrays for the variable simulation parameters are of the same length.
        reference_length = len(
            param_mean_dict
        )  # Use the mean dictionary as the reference.
        for dict_name, d in zip(
            [
                "param_sigma_dict",
                "param_min_dict",
                "param_max_dict",
                "param_transform_dict",
            ],
            [
                param_sigma_dict,
                param_min_dict,
                param_max_dict,
                param_transform_dict,
            ],
        ):
            if len(d) != reference_length:
                raise ValueError(
                    f"Length of {dict_name} does not match length of param_transform_dict. Got {len(d)}, expected {reference_length}."
                )

        # Step 2: Ensure that all keys match for all the parameter arrays.
        reference_keys = (
            param_mean_dict.keys()
        )  # Extract the reference keys from the mean dictionary.
        for dict_name, d in zip(
            [
                "param_sigma_dict",
                "param_min_dict",
                "param_max_dict",
                "param_transform_dict",
            ],
            [
                param_sigma_dict,
                param_min_dict,
                param_max_dict,
                param_transform_dict,
            ],
        ):
            if set(d.keys()) != reference_keys:
                raise ValueError(
                    f"Keys in {dict_name} do not match the reference keys from param_mean_dict. Got {d.keys()}, expected {reference_keys}."
                )

        # Step 3: Ensure that the types of the arguments in each array are correct.
        # Check if values are floats for mean, std, min and max dictionaries.
        for dict_name, d in zip(
            [
                "param_mean_dict",
                "param_sigma_dict",
                "param_min_dict",
                "param_max_dict",
            ],
            [
                param_mean_dict,
                param_sigma_dict,
                param_min_dict,
                param_max_dict,
            ],
        ):
            for key, value in d.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Values in {dict_name} should be numeric floats or ints. Got {type(value)} for key {key}."
                    )

        # For the transform dictionary, check if the values are within the valid transformation types.
        for key, value in param_transform_dict.items():
            if value not in transform_string_int_mapping.keys():
                raise ValueError(
                    f"Valid options for values in param_transform_dict are {transform_string_int_mapping.keys()}. Got {value} for key {key}."
                )

            # TODO: Tentatively lockdown "logitsample", as it is a bit weird and might induce inconsistent batch sizes.
            if value == "logitsample":
                raise NotImplementedError(
                    f"logitsample currently locked due to possibility of inconsistent batch sizes. Found {value} for {key}."
                )

    @staticmethod
    def build_sim_params_arr_var(
        param_mean_dict: Dict[str, float],
        param_sigma_dict: Dict[str, float],
        param_min_dict: Dict[str, float],
        param_max_dict: Dict[str, float],
        param_transform_dict: Dict[str, str],
        transform_string_int_mapping: Dict[int, str],
    ) -> np.ndarray:
        """Builder function that initializes the array of variable simulation parameters to be passed to the GPU execution thread. This should produce the numbers required to sample a `modelPar` array as used in the original codebase.

        This builder only needs to be called once per-run, as it constructs the sampling parameters, not the sampled parameters.
        """

        # Initializing the array based on the number of elements in the input kwargs array.
        # We already validated that the number of keys in each dictionary are identical using the validation function.
        # This means we can initialize a shape based on a reference, in this case, param_mean.
        # This output array will have the shape (params_count, 5)
        # The param-count-axis will be used for vectorized sampling (mean [0]/sigma [1]) and clipping of values (min [2], max [3]) by the sampling function.
        # The transformation axis will also be used to perform the required transformations.
        sim_params_arr_var = np.empty((5, len(param_mean_dict)), dtype=np.float32)

        # Populate the array with the values from the provided kwargs.
        for ind, (mean, sigma, min, max, transform_str) in enumerate(
            zip(
                param_mean_dict.values(),
                param_sigma_dict.values(),
                param_min_dict.values(),
                param_max_dict.values(),
                param_transform_dict.values(),
            )
        ):
            # We need to convert the transform index into the corresponding index value to store it in this array.
            # Based on [None, "log", "logit", "logitsample"]
            transform_index = transform_string_int_mapping[transform_str]
            sim_params_arr_var[:, ind] = np.array(
                [mean, sigma, min, max, transform_index]
            )

        return sim_params_arr_var

    @staticmethod
    def sample_simulation_variables(
        sim_params_arr_var: np.ndarray, total_sequences: int
    ) -> np.ndarray:
        """A vectorized implementation of the sampling logic for simulation variables. Uses the dimensionless parameters to sample from an assumed Gaussian. Reference is `randomParModel` from the original code.

        Args:
            sim_params_arr_var (np.ndarray): The sampling parameters as built by `build_sim_params_arr_var`. Should be an array of the shape (params_count, 5), where the length-5 axis is ordered as [mean, sigma, clip_min, clip_max, transform].
            total_sequences (int): The number of sequences to sample for. This number generally represents the number of parallel simulations to generate simultaneously.

        Returns:
            np.ndarray: The sampled simulation parameters, in the shape (total_sequences, params_count)
        """
        params_count = sim_params_arr_var.shape[1]

        par_mean = sim_params_arr_var[0]
        par_sigma = sim_params_arr_var[1]
        par_clip_min = sim_params_arr_var[2]
        par_clip_max = sim_params_arr_var[3]
        par_transform = sim_params_arr_var[4]

        # Step 1: Sample a uniform random value. This will be used to apply sigma.
        rnd = (2.0 * np.random.rand(total_sequences, params_count) - 1.0).astype(
            np.float32
        )

        # Step 2: Apply sigma as a multiplicative factor. Use vectorized conditional masks.
        # create a non-zero mask, as applying log transforms to zero-values will result in -inf by default.
        non_zero_mask = par_sigma != 0.0
        # Initializing a zero array. This means the default value (hit when the sigma param is 0), will be 0.0.
        sig = np.zeros(params_count, dtype=np.float32)
        # Handle the transformation conditional masks.
        # sig[non_zero_mask] = np.where(
        #     par_transform[non_zero_mask] == 1,
        #     np.log(
        #         par_sigma[non_zero_mask]
        #     ),  # When there is a "log" transform, i.e. transform=1
        #     np.where(
        #         par_transform[non_zero_mask] in [2, 3],
        #         np.log(
        #             par_sigma[non_zero_mask] / (1 - par_sigma[non_zero_mask])
        #         ),  # When there is "logit": 2 of "logitsample": 3 transforms
        #         par_sigma[non_zero_mask],  # When there is no transform.
        #     ),
        # )
        sig[non_zero_mask] = np.where(
            par_transform[non_zero_mask] == 0,  # No transform condition: 0
            par_sigma[non_zero_mask],
            np.where(
                par_transform[non_zero_mask]
                == 1,  # When there is a "log" transform, i.e. transform=1
                np.log(par_sigma[non_zero_mask]),
                np.where(
                    np.isin(par_transform[non_zero_mask], [2, 3]),
                    np.log(
                        par_sigma[non_zero_mask] / (1 - par_sigma[non_zero_mask])
                    ),  # When there is "logit": 2 of "logitsample": 3 transforms
                    0,
                ),
            ),
        )
        # sig = np.broadcast_to(sig, shape=(total_sequences, *sig.shape))
        rnd = rnd * sig

        # Step 3: Add the mean value.
        # Apply vectorized transformations for the second block
        # Apply transformations using vectorized operations
        # No transform condition
        # rnd += np.where(par_transform == 0, par_mean, 0)
        # # "log" transformation
        # rnd += np.where(par_transform == 1, np.exp(np.log(par_mean)), 0)
        # # When there is "logit": 2 of "logitsample": 3 transforms
        # rnd += np.where(np.isin(par_transform, [2, 3]), np.exp(np.log(par_mean / (1 - par_mean))), 0)
        # rnd /= np.where(np.isin(par_transform, [2, 3]), 1.0 + rnd, 1)

        rnd += np.where(
            par_transform == 0,  # No transform condition: 0
            par_mean,
            np.where(
                par_transform == 1,  # When there is a "log": 1 transform
                np.log(par_mean),
                np.where(
                    np.isin(
                        par_transform, [2, 3]
                    ),  # When there is "logit": 2 of "logitsample": 3 transforms
                    np.log(par_mean / (1 - par_mean)),
                    0,
                ),
            ),
        )
        # In the case of log transforms [log (1), logit (2), logitandsample (3)], exponentiate the value prior to clipping.
        rnd[:, np.isin(par_transform, [1, 2, 3])] = np.exp(
            rnd[:, np.isin(par_transform, [1, 2, 3])]
        )

        rnd /= np.where(np.isin(par_transform, [2, 3]), 1.0 + rnd, 1)

        # Clip values
        rnd = np.clip(rnd, a_min=par_clip_min, a_max=par_clip_max)

        return rnd
