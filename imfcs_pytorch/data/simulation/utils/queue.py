"""Implementation of a simulation queue. Not necessarily the most efficient, but written in a way which more easily integrates with the concept of a PyTorch Dataset."""

import numpy as np
import queue
import multiprocessing
import time

# Typing specific imports
from typing import Tuple


# See https://numba.discourse.group/t/cuda-how-to-run-concurrently-kernels-using-multiprocessing/1538/12 for examples of how to trigger multiprocessing for CUDA kernels, and how interleaving might affect things
class SimulationQueue:
    def __init__(
        self,
        simulation_function: callable,
        simulation_parameter_function: callable,
        total_simulations: int,
        max_queue_size: int = 120000,
        retry_time: int = 3,
    ):
        """Simple wrapper to get a simulation queue that can be used in a Dataset, complete with blocking behaviour for multithreaded execution.

        To use:
        ```python
        sim_queue = SimulationQueue(
            simulation_function,    # The simulation function.
            simulation_parameter_function,      # Function to produce simulation parameters as a kwargs dictionary, to be called every simulation run.
            total_simulations,  # Number of simulations before stopping.
            max_queue_size,     # The maximum size of the queue, defaults to 120000. Set to 0 for infinite queue size (strongly discouraged, as memory use could explode).
        )
        # Start simulation threads in a separate thread
        # This needs to be in a separate thread, else will block the main thread.
        simulation_thread = threading.Thread(target=sim_queue)
        simulation_thread.start()

        # Get items from the queue by popping.
        result = sim_queue.get_result_from_queue()
        print(f"Queue size: {sim_queue.simulation_queue.qsize()}")
        ```

        Args:
            simulation_function (callable): Simulation function to execute.
            simulation_parameter_function (callable): Function to get the parameters expected by the simulation.
            total_simulations (int, optional): Total number of simulations before stopping the simulation worker.
            max_queue_size (int): The maximum size of the simulations queue, i.e. the maximum number of simulations to store before blocking. Can be set to None for an infinite queue, but not recommended as memory use can explode due to the fast simulation speed. Defaults to 120000.
            retry_time (int, optional): Wait time before attempting a retry for queue insertion or queue extraction. Defaults to 3.
        """
        # Initializing the queue
        # A multiprocessing queue is used here, which allows for PyTorch dataloaders to utilize multiple workers.
        self.simulation_queue = multiprocessing.Queue(maxsize=max_queue_size)
        self.total_simulations = total_simulations
        self.simulation_count = 0
        # self.lock = threading.Lock()

        # Storing the simulation function and the method for sampling parameters.
        # Note that the simulation_function takes *args and **kwargs, so order matters for the implementation of the simulation parameter function.
        self.simulation_function = simulation_function
        self.simulation_parameter_function = simulation_parameter_function

        self.retry_time = retry_time

    def run_simulation(self, **kwargs) -> np.ndarray:
        """Execute the simulation function and return the simulated image stack.

        Returns:
            np.ndarray: Simualted image stack.
        """
        # Instead of using kwargs as a dict, cast to a tuple, as the cuda compile code assumes no specific names.
        # Leverage the fact that Python dictionaries are insert-ordered.
        casted_args = tuple(kwargs.values())
        simulated_stack = self.simulation_function(*casted_args)

        return simulated_stack

    def simulation_worker(self, simulations_started_signal_func: callable):
        """Simulation worker to be called via multithreading.

        Args:
            simulations_started_signal_func (callable): A signal function which allows the threads to be terminated, avoiding the problems of zombie threads. Provided via the interface, and set to False once simulations are sufficient and terminated.
        """
        # TODO: This condition seems to never trigger. Doesn't really matter much since the Dataset itself handles the stopping through a range-based iterator, but should have additional safety.
        while self.simulation_count < self.total_simulations:
            print(
                f"{self.simulation_count} < {self.total_simulations} - Terminate={not (self.simulation_count < self.total_simulations)}"
            )
            # Get the next simulation parameters
            # We also specifically return the target_array from the parameter generation function
            # This is because we want our queues to have the inputs associated with the targets.
            kwargs, target_arr = self.simulation_parameter_function()
            simulated_stack = self.run_simulation(**kwargs)
            # Force stop of the simulations if the signal is sent from the interface.
            if not simulations_started_signal_func():
                break
            while True:
                # Force stop of the simulations if the signal is sent from the interface.
                # Add an additional call here in case the simulator is stuck in a try/retry loop.
                if not simulations_started_signal_func():
                    break
                try:
                    for batch_x, batch_y in zip(simulated_stack, target_arr):
                        self.simulation_queue.put([batch_x, batch_y], block=False)
                        self.simulation_count += 1
                except queue.Full:
                    print("Queue is full, waiting for retry.")
                    # Queue is empty, wait and retry
                    time.sleep(self.retry_time)

    def clear_queue(self):
        """Function to teardown the queue at the end of simulations.

        Function that allows threads to be joined without blocking. See: https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.cancel_join_thread.

        The `cancel_join_thread()` method functions as a `allow_exit_without_flush()` method which destroys the data in the queue, preventing deadlocks.
        """
        self.simulation_queue.cancel_join_thread()

    def get_result_from_queue(self) -> Tuple[np.ndarray, np.ndarray]:
        """Getter function to extract a single element from the queue.

        This function likely integrates with the Dataset object, and extracts a single item from the queue. Any logic specific to PyTorch should only happen in the context of the Dataset.

        Returns:
            Tuple[np.ndarray,np.ndarray]: Training batch in the form of [Simulated Image Stack, Target Array]
        """
        # Get a result from the queue
        while True:
            try:
                return self.simulation_queue.get(block=False)
            except queue.Empty:
                print("Queue empty, waiting for retry.")
                time.sleep(self.retry_time)
