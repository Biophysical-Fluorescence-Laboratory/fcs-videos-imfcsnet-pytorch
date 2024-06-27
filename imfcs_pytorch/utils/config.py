"""A set of helper functions that make working with YACS config files."""


def extract_key_index_from_dict(dictionary_object: dict, key_name_string: str) -> int:
    """Helper function to extract a key's index from a dictionary given it's name string by leveraging the immutable state of YACS dictionaries (provided they are frozen with `freeze()`).

    The starting point for this was the original ImFCSNet code's reliance on Enums, which were a bit inelegant to work with due to the high number of interlinked files and classes. YACS inherently provides some semblance of ordering due to it's use of dictionary-like key/value pairs, which means we can directly parse out the desired indices of values given the key name-string and the original dictionary object.

    This is necessary because the GPU-based simulations do not support dictionaries, meaning the input parameters need to be passed in as arrays, which are then used to simulate across the assigned simulator threads.

    Args:
        dictionary_object (dict): The dictionary object that we are pulling from (YACS config objects inherit from dicts).
        key_name_string (str): The name of the key to extract the index for.

    Raises:
        ValueError: `key_name_string` is not a valid key in `dictionary_object`.

    Returns:
        int: The index of `key_name_string` in `dictionary_object`.
    """
    # Check if the key name exists in the dictionary.
    if key_name_string not in dictionary_object.keys():
        raise ValueError(
            f"key_name_string {key_name_string} is not found in the provided dictionary_object. Available keys are {dictionary_object.keys()}"
        )

    return list(dictionary_object.keys()).index(key_name_string)
