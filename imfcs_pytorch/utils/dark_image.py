"""These cover the functionality involved for extracting the probability mass function from a set of dark images.

To use:
```shell
python imfcs_pytorch/utils/dark_image.py \
    --files path/to/tiff/files/*.tif \
    --out corrected_pmf_2d.npy
```

Then, you can use your dark images' noise profile to handle EMCCD noise generation. Simply replace the pointer to the generated PMF in your config file:
```yaml
TRANSFORMS:
  TRAIN:
    NOISE:
      EMCCD:
        PMF_FILE: path/to/corrected_pmf_2d.npy
```
"""

import numpy as np
from collections import defaultdict

# Typing-specific imports
from typing import Dict, List


def handle_extreme_pixel_vals(img_stack: np.ndarray, ceiling: int) -> np.ndarray:
    """We notice that there are cases where the pixel values reach extreme numbers (16k and up). These have been confirmed to be unwanted, and we decided to zero out these pixels.

    Args:
        img_stack (np.ndarray): Input image stack (likely a dark image).
        ceiling (int): The maximum value pixels are expected to take. Anything above this value will be zeroed out.

    Returns:
        np.ndarray: Input image stack with any extreme pixel values zeroed out.
    """
    return np.where(img_stack >= ceiling, 0, img_stack)


def hirsch_correction(
    img_stack: np.ndarray, isotropic_noise_assumption: bool = False
) -> np.ndarray:
    """Conduct Hirsch correction on an input image stack.

    Reference:
    A Stochastic Model for Electron Multiplication Charge-Coupled Devices – From Theory to Practice
    Michael Hirsch, Richard J. Wareham, Marisa L. Martin-Fernandez, Michael P. Hobson, Daniel J. Rolfe


    Args:
        img_stack (np.ndarray): Input image stack (likely a dark image) to do Hirsch correction on.
        isotropic_noise_assumption (bool, optional): Whether to assume that noise is isotropic over the input. If True (Eqn 7), only subtract the overall mean. Defaults to False (Eqn 1), where we subtract the row-wise and column-wise means, in addition to the overall mean.

    Returns:
        np.ndarray: The input dark image stack with Hirsch correction applied.
    """
    # Do correction of image stack
    # If requested, assume that noise is correlated, and subtract the row and column means.
    if not isotropic_noise_assumption:
        _col_component = np.mean(img_stack, axis=1, keepdims=True)
        _row_component = np.mean(img_stack, axis=2, keepdims=True)
        img_stack = img_stack + np.mean(img_stack) - _col_component - _row_component
    else:
        # By default, only remove the mean over all frames.
        # If the noise is uncorrelated and isotropic
        img_stack = img_stack - np.mean(img_stack)

    return img_stack


def get_pixel_val_counts(img_stack: np.ndarray) -> Dict[int, int]:
    """For a given image stack (likely a dark image), calculate the counts of each individual pixel value.

    For the purposes of this function, every value between the minimum and the maximum will be counted (with values that do not exist being assigned the 0 value). This is to allow this to interface with the EMCCD noise generation functions.

    Args:
        img_stack (np.ndarray): Input image stack (likely a dark image, preferably one that has been Hirsch corrected).

    Returns:
        Dict[int, int]: Dictionary with the key:value mapping of {pixel_val: count}
    """
    pixel_val_count_dict = {}

    # Populate the pixel_vals first
    # We do this first and in an inelegant way, but this is to ensure that all pixel values exist in our dict.
    # For example, if -30 and -28 exist in our images, but -29 does not, our probability mass function would not have a representation of -29.
    # Thus, we populate every valid integer value between the minimum and the maximum
    _vals = np.arange(np.amin(img_stack), (np.amax(img_stack) + 1))
    for _val in _vals:
        # Only create the entry if it does not already exist.
        if _val not in pixel_val_count_dict.keys():
            pixel_val_count_dict[_val] = 0

    # Finally, update the counts.
    pixel_vals, counts = np.unique(img_stack, return_counts=True)
    for pixel_val, count in zip(pixel_vals, counts):
        pixel_val_count_dict[pixel_val] += count

    return pixel_val_count_dict


def add_dicts(list_of_dicts: List[Dict[int, int]]) -> Dict[int, int]:
    """Takes a list of dictionaries and adds them together.

    Args:
        list_of_dicts (List[Dict[int, int]]): List of dictionaries with the key:value mapping of {pixel_val: count}

    Returns:
        Dict[int, int]: Single dictionary with the key:value mapping of {pixel_val: count}, summed across the input list.
    """
    final_dict = defaultdict(int)
    for dct in list_of_dicts:
        for key, value in dct.items():
            final_dict[key] += value
    return final_dict


def convert_dict_of_counts_to_pmf(dict_of_counts: Dict[int, int]) -> np.ndarray:
    """Takes a dictionary of counts, and converts it into a NumPy array of the probability mass function.

    Args:
        dict_of_counts (Dict[int, int]): Dictionary with the key:value mapping of {pixel_val: count}.

    Returns:
        np.ndarray: NumPy array of the probability mass function, where counts have been converted into proportions/probabilities.
    """
    # To convert the counts to probabilities, we divide it by the sum
    probability_mass_function = np.array(list(dict_of_counts.values()))
    probability_mass_function = probability_mass_function / np.sum(
        probability_mass_function
    )  # Normalize by the sum

    return probability_mass_function


def save_pmf_as_npy(
    save_fp: str, dict_of_counts: Dict[int, int], probability_mass_function: np.ndarray
):
    """Save the PMF in a format compatible with the currently employed data augmentation API.

    We need to append a single int value to the end of the array. This is because our probability distributions do not start at 0, so we need to add the minimum value to out sampling process to recover the exact distribution.

    Args:
        save_fp (str): File path to save the NumPy array to.
        dict_of_counts (Dict[int, int]): Dictionary with the key:value mapping of {pixel_val: count}. Required to extract the
        probability_mass_function (np.ndarray): _description_
    """
    np.save(
        save_fp,
        np.concatenate(
            [
                probability_mass_function,
                [list(dict_of_counts.keys())[0]],
            ]
        ),
    )


if __name__ == "__main__":
    """Utility to loop through a list of dark image TIFF files and generate the corresponding PMF files for simulated EMCCD noise generation."""
    import os
    import argparse
    import tifffile
    import glob

    def get_parser() -> argparse.ArgumentParser:
        """Creates the parser.

        Returns:
            argparse.ArgumentParser: CLI Argument Parser.
        """
        parser = argparse.ArgumentParser(
            description="Derivation of PMF from dark image."
        )
        parser.add_argument(
            "--files",
            required=True,
            default=None,
            metavar="FILE",
            nargs="+",
            help="Paths to the dark image TIF files to derive the PMF from. To derive based on multiple files, provide a space-separated list of files. For example, `--files ./1.tif ./2.tif` will operate on both tiff files. Similarly, glob-based extensions are supported. For example, to run inference on all files with the `.tif` extension in a folder, you can use `--files ./folder/*.tif`",
        )
        parser.add_argument(
            "--out",
            required=True,
            type=str,
            metavar="FILE",
            help="Path to the output where the PMF will be saved. The PMF will be in the form of a NumPy array, so using the '.npy' extension is recommended.",
        )
        parser.add_argument(
            "--ceil",
            required=False,
            type=str,
            default=None,
            help="The maximum value pixels are expected to take. Anything above this value will be zeroed out.",
        )
        parser.add_argument(
            "--hirsch-isotropic",
            required=False,
            type=bool,
            default=False,
            help="Whether to assume that noise is isotropic over the input. If True (Eqn 7), only subtract the overall mean. Defaults to False (Eqn 1), where we subtract the row-wise and column-wise means, in addition to the overall mean.",
        )

        return parser

    def parse_list_of_files(filepath_args: str) -> List[str]:
        list_of_files = []

        for filepath in filepath_args:
            if "*" in filepath:
                list_of_files.extend(glob.glob(filepath))
            else:
                list_of_files.append(filepath)

        if len(list_of_files) == 0:
            raise ValueError(
                "Parsed list of files has length of 0. There are no valid files to execute inference on."
            )
        return list_of_files

    # Parse arguments from the CLI.
    args = get_parser().parse_args()

    arg_files = args.files
    arg_out = args.out
    arg_ceil = args.ceil
    arg_hirsch_isotropic = args.hirsch_isotropic

    if os.path.isdir(os.path.dirname(arg_out)):
        if not os.path.exists(os.path.dirname(arg_out)):
            raise ValueError(
                f"Output directory {os.path.dirname(arg_out)} does not exist."
            )

    # Loop through all of the provided files.
    file_paths = parse_list_of_files(arg_files)

    # Keep track of the extracted count directories.
    list_of_count_dicts = []
    for fp in file_paths:
        print(f"Processing {fp}")
        # Read the TIFF file
        arr = tifffile.imread(fp)

        if arg_ceil is not None:
            arr = handle_extreme_pixel_vals(arr, ceiling=arg_ceil)

        # Do Hirsch correction, and cast to integer.
        arr = np.round(
            hirsch_correction(arr, isotropic_noise_assumption=arg_hirsch_isotropic)
        ).astype(int)

        # Do counting, and add to the list for tracking.
        list_of_count_dicts.append(get_pixel_val_counts(arr))

    # Sum all of the dictionaries together.
    pixel_val_count_dict = add_dicts(list_of_count_dicts)

    # Derive PMF from the counts dictionary.
    pmf = convert_dict_of_counts_to_pmf(pixel_val_count_dict)

    # Save the file.
    save_pmf_as_npy(arg_out, pixel_val_count_dict, pmf)
