"""Bleaching is innevitable in physical samples, and applying some form of bleach correction is standard practice for most evaluation of experimental samples. We implement our own variation of the polynomial bleach correction scheme here, and it comes within 0.0005 of the original Java curve fitters (https://github.com/shaoren-sim/imfcsnet-pytorch/issues/5#issuecomment-1968173667).

The fits will likely never be perfect with Java's math3-based fits, as those have an undocumented curve fitting algorithm. However, this can be considered to be close enough.

The main implementation is heavily vectorized, such that it can operate over intensity trace batches effectively. However, the original unvectorized implementation is also included for reference purposes. Just note that those are unused for a reason, as they take more than 5 times the required time for inference.

"""

import numba
import numpy as np
from scipy.optimize import curve_fit

# Typing-specific imports


class PolynomialFitVectorized:
    def __init__(self, poly_order: int):
        """This function attempts to streamline the fitting process by performing a vectorized fit for a significant speedup (20 seconds compared to 15 minutes for a 60000x128x128 input).

        This does sacrifice the ability to use initial guesses, so that is worth bearing in mind.

        Args:
            poly_order (int): Number of polynomial factors to include.
        """
        self.poly_order = poly_order

    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
        """Perform fitting using NumPy's interface to MINPACK.

        See https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyfit.html for more details.

        Args:
            x_data (np.ndarray): Time points, generally obtained through `calc_time_points()`
            y_data (np.ndarray): The averaged intensity traces corresponding to each time point.

        Returns:
            np.ndarray: Fitted coefficients for each coefficient.
        """
        # Note that here, we do not have access to an initial guess parameter.
        coefficients = np.polynomial.polynomial.polyfit(
            x_data, y_data, deg=self.poly_order
        )

        return coefficients


@numba.njit
def calc_intensity_trace_vect(input_signal: np.ndarray, ave_stride: int) -> np.ndarray:
    """Calculate the averaged intensity trace to use for fitting to the polynomial.

    While it is possible to fit the full intensity trace, that would be computationally heavy. Instead, we follow the Java implementation by using an averaged representation of the intensity trace. This is calculated as `frames // ave_stride`, where `ave_stride` is the windowing stride.

    Args:
        input_signal (np.ndarray): Full intensity trace. For the vectorized implementation here, assumes that it is batched, i.e. shape = (FRAMES, NO_OF_PIXELS).
        ave_stride (int): Number of points to use per each window.

    Returns:
        np.ndarray: Averaged intensity trace, will be of shape (FRAMES // ave_stride, NO_OF_PIXELS)
    """
    # Parse out the shape of the input, as these values are constantly in use.
    frames, pixels = input_signal.shape

    # Generates the intensity traces for a given input image.
    # The intensity traces are required for purposes of bleach correction.
    no_of_points = frames // ave_stride  # no_frames / ave_stride

    # Initialize return intensity trace
    int_trace = np.zeros(shape=(no_of_points, pixels))

    # Perform summation for this pixel position
    for p in range(no_of_points):
        _start = p * ave_stride
        _end = (p + 1) * ave_stride
        sum1 = np.sum(input_signal[_start:_end], axis=0)
        int_trace[p] += sum1 / ave_stride

    return int_trace


@numba.njit
def calc_time_points(
    input_stack: np.ndarray, ave_stride: int, frame_time: float
) -> np.ndarray:
    """Calculate the time points after averaging.

    Can be considered to be the x-axis counterpart to the average intensity trace calculated from `calc_intensity_trace_vect`.

    Args:
        input_stack (np.ndarray): Image stack of shape (FRAMES, WIDTH, HEIGHT).
        ave_stride (int): Number of points to use per each window.
        frame_time (float): The frame time used during acquisition.

    Returns:
        np.ndarray: The time points in shape (FRAMES // ave_stride).
    """
    # Parse out the shape of the input, as these values are constantly in use.
    frames, _, _ = input_stack.shape

    # Generates the intensity traces for a given input image.
    # The intensity traces are required for purposes of bleach correction.
    no_of_points = frames // ave_stride  # no_frames / ave_stride

    # Intialize and populate the return time points
    int_time = np.empty(no_of_points)
    for p in range(no_of_points):
        int_time[p] = frame_time * (p + 0.5) * ave_stride

    return int_time


@numba.njit
def correct_intensities(
    input_trace: np.ndarray, poly_coeffs: np.ndarray, frame_time: float
) -> np.ndarray:
    """Correct intensities using the fitted polynomial coefficients.

    This is written to handle vectorized cases, and simultaneously applies the polynomial coefficients over every single pixel simultaneously for a faster bleach correction implementation.

    Args:
        input_trace (np.ndarray): Uncorrected input intensity trace. Is assumed to be batched in the format (FRAMES, WIDTH*HEIGHT)
        poly_coeffs (np.ndarray): Fitted polynomial coefficients.
        frame_time (float): The frame time used during acquisition.


    Returns:
        np.ndarray: Corrected intensity traces. Retains the batched format with shape (FRAMES, WIDTH*HEIGHT)
    """
    frames = input_trace.shape[0]

    corrected_intensities = np.zeros(input_trace.shape)

    for t in range(frames):
        cor_func = np.zeros(poly_coeffs.shape[1])
        for i in range(len(poly_coeffs)):
            cor_func += poly_coeffs[i] * np.power(frame_time * (t + 0.5), i)

        corrected_intensities[t,] = input_trace[t] / np.sqrt(
            cor_func / poly_coeffs[0]
        ) + poly_coeffs[0] * (1 - np.sqrt(cor_func / poly_coeffs[0]))

    return corrected_intensities


def fit_and_correct_wrapper(
    input_stack_trace: np.ndarray,
    ave_stride: int,
    time_points: np.ndarray,
    fit_func: callable,
    frame_time: float,
) -> np.ndarray:
    """Convenience function to conduct preprocessing, fitting and bleach correction in a single function call.

    Args:
        input_stack_trace (np.ndarray): Uncorrected input intensity trace. Is assumed to be batched in the format (FRAMES, WIDTH*HEIGHT)
        ave_stride (int): Number of points to use per each window.
        time_points (np.ndarray): Averaged time points in shape (FRAMES // ave_stride). Calculated via `calc_time_points()` externally as all intensity traces share the same x-axis fit conditions.
        fit_func (callable): Fit function to use for fitting purposes. In the context of this module, this is the polynomial fit.
        frame_time (float): The frame time used during acquisition.

    Returns:
        np.ndarray: Corrected intensity traces in a batch of shape (FRAMES, WIDTH*HEIGHT)
    """
    intensity_trace = calc_intensity_trace_vect(
        input_stack_trace, ave_stride=ave_stride
    )

    # This line seems to be necessary to get a good match with ImagingFCS v1.613.
    # This has been confirmed to NOT be intended behaviour.
    # For now, this line is left here for documentation purposes.
    # intensity_trace = np.floor(intensity_trace)

    poly_coeffs = fit_func(time_points, intensity_trace)

    # Finally, correct the original intensity trace.
    return correct_intensities(input_stack_trace, poly_coeffs, frame_time)


class PolynomialBleachCorrectionModule:
    def __init__(
        self,
        poly_order: int,
        frame_time: float,
        ave_stride: int = 50,
        binning_x_dim: int = 1,
        binning_y_dim: int = 1,
    ):
        """Module handling the logic of polynomial bleach correction.

        This should be the entry point for any bleach correction calls, as it wraps the logic of pre-processing, polynomial fitting and bleach correction in a single convenience class.

        Args:
            poly_order (int): umber of polynomial factors to include. See https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyfit.html for implementation details.
            frame_time (float): The frame time used during acquisition.
            ave_stride (int, optional): Number of points to use per each window.. Defaults to 50.
            binning_x_dim (int, optional): Number of pixels to use for binning over the x-dimension. Defaults to 1.
            binning_y_dim (int, optional): Number of pixels to use for binning over the y-dimension. Defaults to 1.
        """
        self.poly_fit = PolynomialFitVectorized(poly_order)
        self.poly_order = poly_order
        self.frame_time = frame_time
        self.ave_stride = ave_stride

        self.binning_x_dim = binning_x_dim
        self.binning_y_dim = binning_y_dim

    def correct_full_stack(self, image_stack: np.ndarray) -> np.ndarray:
        """Correct the full image stack.

        Wraps the logic of pre-processing, polynomial fitting and bleach correction in a single method call.
        Args:
            image_stack (np.ndarray): Input image stack. This is assumed to be directly read from a TIFF file, in the shape (FRAMES, WIDTH, HEIGHT).

        Returns:
            np.ndarray: Image stack with bleach correction applied, in the shape (FRAMES, WIDTH, HEIGHT).
        """
        # Parse out the shape of the input, as these values are constantly in use.
        frames, shape_x, shape_y = image_stack.shape

        # Get the background value.
        # Use the most basic assumption that the background is the lowest value.
        image_stack = image_stack - np.amin(image_stack)  # Correct for the background

        # As per the original Java code, we need to extract the average intensity trace.
        # However, we wrap the average intensity trace extraction in a vectorizable function.
        # We extract the time points first, since these can be reused.
        time_points = calc_time_points(
            input_stack=image_stack,
            ave_stride=self.ave_stride,
            frame_time=self.frame_time,
        )

        # Initially, we performed a fit for
        # Use the extracted intensity trace to fit the polynomial.
        # for x in range(shape_x):
        #     for y in range(shape_y):
        #         poly_coeffs = self.poly_fit.fit(time_points, intensity_trace[:, x, y])

        #         # Finally, correct the original intensity trace.
        #         corrected_image_stack[:, x, y] = correct_trace(image_stack[:, x, y], poly_coeffs, self.frame_time)

        reshaped_input = image_stack.reshape(frames, shape_x * shape_y)

        # Use this bottom block to verify that the pre-correction values match the Java code.
        # This should be a perfect match since we are reading the same tiff with no further processing.
        # Note that the ImageJ plugin averages every 50 points, and casts to int
        # print(reshaped_input[:, 0])
        # print(np.mean(reshaped_input[:, 0].reshape((-1, 50)), axis=1).astype(int))
        # import matplotlib.pyplot as plt
        # plt.plot(np.arange(frames // 50), np.mean(reshaped_input[:, 0].reshape((-1, 50)), axis=1).astype(int))
        # plt.savefig("pre_with_avg.png")
        # raise Exception("Verify pre-correct")

        # corrected_image_stack = np.apply_along_axis(
        #     fit_and_correct_wrapper,
        #     axis=0,
        #     arr=image_stack,
        #     fit_func=self.poly_fit.fit,
        #     ave_stride=self.ave_stride,
        #     time_points=time_points,
        #     frame_time=self.frame_time,
        # )

        corrected_image_stack = fit_and_correct_wrapper(
            reshaped_input,
            fit_func=self.poly_fit.fit,
            ave_stride=self.ave_stride,
            time_points=time_points,
            frame_time=self.frame_time,
        )

        # Use this bottom block to verify that the post-correction values match the Java code.
        # At this current stage, the values do not match perfectly, with slightly higher values that are <1.0 higher than Java.
        # This is currently deemed to be 'good enough', especially given the speed tradeoff if applying more involved fitting algorithms.
        # corrected_avg_trace = debug_fit_and_correct_avg_trace(
        #     reshaped_input,
        #     fit_func=self.poly_fit.fit,
        #     ave_stride=self.ave_stride,
        #     time_points=time_points,
        # )

        # print(corrected_avg_trace[:, 0])
        # print(corrected_avg_trace[:, 0].shape)
        # import matplotlib.pyplot as plt

        # plt.plot(
        #     time_points,
        #     corrected_avg_trace[:, 0],
        # )
        # plt.savefig("post_avg_trace.png")

        # import pandas as pd
        # df = pd.DataFrame(
        #     {
        #         "time": time_points,
        #         "np_bleachcorr": corrected_avg_trace[:, 0],
        #     }
        # )
        # df.to_csv("numpy_corrected_avg_trace.csv")
        # raise Exception("Verify correction of averaged trace to match Java")

        return corrected_image_stack.reshape(frames, shape_x, shape_y)


# Below are the unvectorized implementations. Use these for reference, but just note that these are way slower than the vectorized implementations,
class PolynomialFit:
    """A basic reimplementation of the Java-based fitter."""

    def __init__(self, poly_order):
        self.poly_order = poly_order

    def polynomial(self, x, *coefficients):
        return sum(coefficients[i] * x**i for i in range(self.poly_order + 1))

    def fit(self, x_data, y_data):
        initial_guess = np.zeros(self.poly_order + 1)
        initial_guess[0] = y_data[-1]  # Use the last point as offset estimate

        coefficients, _ = curve_fit(self.polynomial, x_data, y_data, p0=initial_guess)
        return coefficients


@numba.njit
def calc_intensity_trace(
    input_stack: np.ndarray, ave_stride: int, binning_x: int, binning_y: int
) -> np.ndarray:
    # Parse out the shape of the input, as these values are constantly in use.
    frames, shape_x, shape_y = input_stack.shape

    # Generates the intensity traces for a given input image.
    # The intensity traces are required for purposes of bleach correction.
    no_of_points = frames // ave_stride  # no_frames / ave_stride

    # Initialize return intensity trace
    int_trace = np.zeros(shape=(no_of_points, shape_x, shape_y))

    # Perform summation for each pixel
    for ipx in range(
        shape_x - binning_x + 1
    ):  # Loop through possible starting x positions
        for ipy in range(
            shape_y - binning_x + 1
        ):  # Loop through possible starting y positions
            # Perform summation for this pixel position
            for p in range(no_of_points):
                _start = p * ave_stride
                _end = (p + 1) * ave_stride
                sum1 = np.sum(
                    input_stack[
                        _start:_end, ipx : ipx + binning_x, ipy : ipy + binning_y
                    ]
                )
                int_trace[p, ipx, ipy] += sum1 / ave_stride

    return int_trace


@numba.njit
def debug_correct_avg_trace(
    input_trace: np.ndarray, poly_coeffs: np.ndarray, avg_times: np.ndarray
):
    """Function used for debugging. This was made under the observation that the plots in the Java code do not average the corrected 50k frame intensities, but rather, plot the corrected averaged traces."""
    points = input_trace.shape[0]

    corrected_avg_trace = np.zeros(input_trace.shape)

    for t in range(points):
        cor_func = np.zeros(poly_coeffs.shape[1])
        for i in range(len(poly_coeffs)):
            cor_func += poly_coeffs[i] * np.power(avg_times[t], i)

        corrected_avg_trace[t,] = input_trace[t] / np.sqrt(
            cor_func / poly_coeffs[0]
        ) + poly_coeffs[0] * (1 - np.sqrt(cor_func / poly_coeffs[0]))

    return corrected_avg_trace


def debug_fit_and_correct_avg_trace(
    input_stack_trace: np.ndarray,
    ave_stride: int,
    time_points: np.ndarray,
    fit_func: callable,
) -> np.ndarray:
    """Entry point for debugging versus Java implementation. This was made under the observation that the plots in the Java code do not average the corrected 50k frame intensities, but rather, plot the corrected averaged traces."""
    intensity_trace = calc_intensity_trace_vect(
        input_stack_trace, ave_stride=ave_stride
    )
    # This line seems to be necessary to get a good match with ImagingFCS v1.613.
    # This has been confirmed to NOT be intended behaviour.
    # For now, this line is left here for documentation purposes.
    # intensity_trace = np.floor(intensity_trace)

    poly_coeffs = fit_func(time_points, intensity_trace)

    # Finally, correct the original intensity trace.
    return debug_correct_avg_trace(intensity_trace, poly_coeffs, time_points)


if __name__ == "__main__":
    poly_bleach_correction = PolynomialBleachCorrectionModule(4, 0.00206)

    # Generate dummy image stack
    image_stack = np.random.randint(0, 255, (50000, 50, 50))

    poly_bleach_correction.correct_full_stack(image_stack=image_stack)
