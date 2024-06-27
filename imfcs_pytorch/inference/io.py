import os
import torch
import tifffile
import pandas as pd

from typing import List


def process_output_map(
    output_path: str,
    output_map: torch.Tensor,
    output_format: str,
    taget_params_list: List[str],
):
    """Saves the output map in a format conducive for further analysis.

    The model output maps as NumPy arrays are not that easy to process during data analysis. This function allows these arrays to be saved in formats that can be easily processed in Pandas/Excel (CSV) or ImageJ/Fiji (TIFF).

    Args:
        output_path (str): Path to save outputs of desired formats.
        output_map (torch.Tensor): Output map of shape (TEMPORAL_WINDOWS, REGRESSION_TARGETS, WIDTH, HEIGHT)
        output_format (str): Expected output format. Supports 'tif' and 'csv'
        taget_params_list (List[str]): Regression target list. Generally defined from the YACS config file as TASK.REGRESSION.TARGETS.

    Raises:
        ValueError: Invalid output format.
    """
    if output_format == "tif":
        output_map_as_tif(output_path, output_map, taget_params_list)
    elif output_format == "csv":
        output_map_as_csv(output_path, output_map, taget_params_list)
    elif output_format == "all":
        # In the case of "all", we need to reformat the output path.
        _output_path_base_no_ext = os.path.splitext(output_path)[0]
        output_map_as_tif(
            f"{_output_path_base_no_ext}.tif", output_map, taget_params_list
        )
        output_map_as_csv(
            f"{_output_path_base_no_ext}.csv", output_map, taget_params_list
        )
    else:
        raise ValueError(
            f"--output-format must be in ['csv', 'tif', 'all'], got {output_format}."
        )


def output_map_as_tif(
    output_path: str, output_map: torch.Tensor, taget_params_list: List[str]
):
    """Saves the output map as a TIFF file, designed for easier analysis in software like ImageJ and Fiji.

    Since TIFFs are most easily readable when they are 3-dimensional (CHANNELS/TIME, WIDTH, HEIGHT), we do splitting over multiple files when there are multiple regression targets.

    Args:
        output_path (str): Path to save the output tif file.
        output_map (torch.Tensor): Output map of shape (TEMPORAL_WINDOWS, REGRESSION_TARGETS, WIDTH, HEIGHT)
        taget_params_list (List[str]): Regression target list. Generally defined from the YACS config file as TASK.REGRESSION.TARGETS.
    """
    # Get the shape of the tensor
    shape = output_map.shape

    # This case covers the situation where we perform averaging across all chunks, i.e. (C, X, Y) - where C is the number of predicted values.
    if len(shape) == 3:
        tifffile.imwrite(output_path, output_map)
    # This covers the case where there is no temporal windowing (temporal_windowing_strategy == None)
    # In this case, there is an additional chunk index column
    # Since 4D tifs are not entirely supported, we perform spliting across the different predicted values to produce multiple files per output map.
    elif len(shape) == 4:
        for ind, target_param_name in enumerate(taget_params_list):
            _parts = os.path.split(output_path)
            tifffile.imwrite(
                os.path.join(
                    _parts[0], f"_{target_param_name}".join(os.path.splitext(_parts[1]))
                ),
                output_map[:, ind],
            )


def output_map_as_csv(
    output_path: str, output_map: torch.Tensor, taget_params_list: List[str]
):
    """Saves the output map as a CSV file, designed for easier analysis in software like Excel or libraries like Pandas/Polar.

    To ensure maximal information is retained for possible reconstruction of the output maps, the CSV outputs will retain the x- and y-axis coordinates.

    Args:
        output_path (str): Path to save the output CSV file.
        output_map (torch.Tensor): Output map of shape (TEMPORAL_WINDOWS, REGRESSION_TARGETS, WIDTH, HEIGHT)
        taget_params_list (List[str]): Regression target list. Generally defined from the YACS config file as TASK.REGRESSION.TARGETS.
    """
    # Convert the output map of tensors into a map consisting of
    # Get the shape of the tensor
    shape = output_map.shape

    # This case covers the situation where we perform averaging across all chunks, i.e. (C, X, Y) - where C is the number of predicted values.
    if len(shape) == 3:
        # Reshape the tensor to 2D where the first dimension is flattened
        reshaped_output = output_map.reshape(shape[0], -1)

        # Create a list to store column names
        pred_col_names = []
        for i in range(reshaped_output.shape[0]):
            pred_col_names.append(taget_params_list[i])

        # Infer x_coords and y_coords from the shape of the output tensor
        x_coords, y_coords = torch.meshgrid(
            torch.arange(shape[1]), torch.arange(shape[2]), indexing="xy"
        )
        x_coords = x_coords.flatten().numpy()
        y_coords = y_coords.flatten().numpy()

        # Extract predicted values from the reshaped tensor
        predicted_values = reshaped_output.T

        # Create a DataFrame
        df = pd.DataFrame(
            {
                "x_coord": x_coords,
                "y_coord": y_coords,
                **{
                    pred_col_names[i]: predicted_values[:, i]
                    for i in range(len(pred_col_names))
                },
            }
        )
    # This covers the case where there is no temporal windowing (temporal_windowing_strategy == None)
    # In this case, there is an additional chunk index column
    elif len(shape) == 4:
        # Reshape the tensor to 2D where the first dimension is flattened
        reshaped_output = output_map.reshape(shape[0], shape[1], -1)

        # Create a list to store column names
        pred_col_names = []
        for i in range(reshaped_output.shape[1]):
            pred_col_names.append(taget_params_list[i])

        # Infer x_coords and y_coords from the shape of the output tensor
        x_coords, y_coords, t_coords = torch.meshgrid(
            torch.arange(shape[-2]),
            torch.arange(shape[-1]),
            torch.arange(shape[0]),
            indexing="xy",
        )
        t_coords = t_coords.flatten().numpy()
        x_coords = x_coords.flatten().numpy()
        y_coords = y_coords.flatten().numpy()

        # Extract predicted values from the reshaped tensor
        predicted_values = reshaped_output.T

        # Construct the prediction column values.
        pred_col_dict = {}
        for ind, pred_col_name in enumerate(pred_col_names):
            col_vals = predicted_values[:, ind]
            pred_col_dict[pred_col_name] = col_vals.flatten()

        # Create a DataFrame
        df = pd.DataFrame(
            {
                "chunk_ind": t_coords,
                "x_coord": x_coords,
                "y_coord": y_coords,
                **{k: v for k, v in pred_col_dict.items()},
            }
        )

    # Write DataFrame to CSV
    df.to_csv(output_path, index=False)
