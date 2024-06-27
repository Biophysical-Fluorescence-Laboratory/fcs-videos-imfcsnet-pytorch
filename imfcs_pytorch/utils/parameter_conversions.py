"""This file includes the functionality to cast the derived dimensionless parameters of the original paper into physical parameters. These can be used as convenience functions to derive new parameter settings.

There are also convenience functions to cast the inconvenient mod/sig abstractions of the original paper into more convenient, readable and faster min/max implementations."""

import numpy as np

from typing import Tuple


def get_minmax_from_modsig(
    mod: float, sig: float, scale: str = "natural"
) -> Tuple[float, float]:
    """Convert 'mod/sig' representations from the original paper parameter representations to min/max.

    Args:
        mod (float): The mean/median of the expected uniform distribution in natural scale.
        sig (float): The distance from the mean/median of an expected uniform distribution in natural scale to the upper and lower bounds.
        scale (str, optional): Whether to use a log or natural uniform distribution. Defaults to "natural".

    Raises:
        ValueError: Invalid scale parameter.

    Returns:
        Tuple[float, float]: Min/max value in the form (min, max)
    """
    if scale == "natural":
        minval = mod - sig
        maxval = mod + sig

        return minval, maxval
    elif scale == "log":
        minval = np.exp(np.log(mod) - np.log(sig))
        maxval = np.exp(np.log(mod) + np.log(sig))

        return minval, maxval
    else:
        raise ValueError(f'scale must be ["natural", "log"], got {scale}')


def get_modsig_from_minmax(
    minval: float, maxval: float, scale: str = "natural"
) -> Tuple[float, float]:
    """Convert min/max parameter sampling parameters to the 'mod/sig' representations from the original paper parameter representations.

    Args:
        minval (float): Minimum value in natural scale.
        maxval (float): Maximum value in natural scale.
        scale (str, optional): Whether to use a log or natural uniform distribution. Defaults to "natural".

    Raises:
        ValueError: Invalid scale parameter.

    Returns:
        Tuple[float, float]: mod.sig value in the form (mod, sig)
    """
    if scale == "natural":
        mod = (maxval + minval) / 2
        sig = (maxval - minval) / 2

        return mod, sig
    elif scale == "log":
        mod = np.exp((np.log(maxval) + np.log(minval)) / 2)
        sig = np.exp((np.log(maxval) - np.log(minval)) / 2)

        return mod, sig
    else:
        raise ValueError(f'scale must be ["natural", "log"], got {scale}')


# %%
# Parameter conversion
# Parameter and symbols (Just to prevent headaches)
# np_step: EMISSION_RATE -> : natural (related to CPS)
# 𝜌 -> particle_density : log (related to N)
# 𝜎_particle -> particle_sig : log (related to d)
# 𝜔_xy -> PHOTON_SIG: natural (related to PSF)

# ccd noise is not included here, as we split out noise to be used as a data augmentation instead.


# %%
# D / PARTICLE_SIG
def convert_particle_sig_to_d(
    particle_sig: np.ndarray,
    frame_time: float,
    steps_per_frame: int,
    pixel_size: float,
    magnification: float,
) -> np.ndarray:
    """Convert the dimensionless 'ParticleSig' into the physical diffusion coefficient (D).

    Args:
        particle_sig (np.ndarray): PARTICLE_SIG as a dimensionless parameter.
        frame_time (float): Frame time used during image acquisition.
        steps_per_frame (int): Number of simulation steps per frame.
        pixel_size (float): The pixel size in um.
        magnification (float): The objective magnification used during image acquisition.

    Returns:
        np.ndarray: D as a physical parameter in um2/s.
    """
    pixel_size = pixel_size / magnification

    return np.power(particle_sig * pixel_size, 2) / (2 * (frame_time / steps_per_frame))


def convert_d_to_particle_sig(
    d: np.ndarray,
    frame_time: float,
    steps_per_frame: int,
    pixel_size: float,
    magnification: float,
) -> np.ndarray:
    """Convert the physical diffusion coefficient (D) into the dimensionless 'ParticleSig' .

    Args:
        d (np.ndarray): D as a physical parameter in um2/s.
        frame_time (float): Frame time used during image acquisition.
        steps_per_frame (int): Number of simulation steps per frame.
        pixel_size (float): The pixel size in um.
        magnification (float): The objective magnification used during image acquisition.

    Returns:
        np.ndarray: ParticleSig as a dimensionless parameter.
    """

    pixel_size = pixel_size / magnification

    return np.sqrt(2 * d * (frame_time / steps_per_frame)) / pixel_size


# %%
# N / PARTICLE_DENSITY
def convert_particle_density_to_n(
    particle_density: np.ndarray, pixels_per_side: int
) -> np.ndarray:
    """Convert 'ParticleDensity' into a more intrepretable N form.

    Args:
        particle_density (np.ndarray): Particle density dimensionless parameter.
        pixels_per_side (int): Number of pixels per-side in a simulation. Note that this needs to include the margin pixels as well to get a proper representation.

    Returns:
        np.ndarray: Number of particles as a physical parameter.
    """
    return np.floor(particle_density * np.power(pixels_per_side, 2)).astype(int)


def convert_n_to_particle_density(
    n: np.ndarray,
    pixels_per_side: int,
) -> np.ndarray:
    """Convert the physical 'N' parameter into the dimensionless 'ParticleDensity' form.

    Args:
        n (np.ndarray): Number of particlesa sa  physical parameter.
        pixels_per_side (int): Number of pixels per-side in a simulation. Note that this needs to include the margin pixels as well to get a proper representation.

    Returns:
        np.ndarray: Particle density as a dimensionless parameter.
    """
    return n / np.power(pixels_per_side, 2)


# %%
# CPS / EMISSION_RATE
def convert_emission_rate_to_cps(
    emission_rate: np.ndarray,
    frame_time: float,
    steps_per_frame: int,
) -> np.ndarray:
    """Convert the dimensionless 'EmissionRate' parameter into the physical photon counts-per-second CPS parameter.

    Args:
        emission_rate (np.ndarray): EmissionRate as a dimensionless parameter.
        frame_time (float): Frame time used during image acquisition.
        steps_per_frame (int): Number of simulation steps per frame.

    Returns:
        np.ndarray: CPS as a physical parameter.
    """
    return emission_rate / (frame_time / steps_per_frame)


def convert_cps_to_emission_rate(
    cps: np.ndarray,
    frame_time: float,
    steps_per_frame: int,
) -> np.ndarray:
    """Convert the physical photon counts-per-second CPS parameter into the dimensionless 'EmissionRate'.

    Args:
        cps (np.ndarray): CPS as a physical parameter.
        frame_time (float): Frame time used during image acquisition.
        steps_per_frame (int): Number of simulation steps per frame.

    Returns:
        np.ndarray: EmissionRate as a dimensionless parameter.
    """
    return cps * (frame_time / steps_per_frame)


# %%
# PHOTON_SIG (omega_xy) to PSF
def convert_photon_sig_to_psf(
    photon_sig: np.ndarray,
    emission_wavelength: float,
    numerical_aperture: float,
    pixel_size: float,
    magnification: float,
) -> np.ndarray:
    """Convert the dimensionless 'PhotonSig' to the physical point spread function (PSF) parameter.

    Args:
        photon_sig (np.ndarray): 'PhotonSig' as a dimensionless parameter.
        emission_wavelength (float): Emission wavelength in nm.
        numerical_aperture (float): Numerical aperture of the image acquistion setup.
        pixel_size (float): The pixel size in um.
        magnification (float): The objective magnification used during image acquisition.

    Returns:
        np.ndarray: PSF as a physical parameter.
    """
    # Assume that the pixel size is provided in micro metres
    pixel_size = pixel_size * 1e-6
    # and that the emission wavelength is provided in nanometres
    emission_wavelength = emission_wavelength * 1e-9

    pixel_size = pixel_size / magnification

    return (photon_sig * pixel_size * 2 * numerical_aperture) / emission_wavelength


def convert_psf_to_photon_sig(
    psf_xy: np.ndarray,
    emission_wavelength: float,
    numerical_aperture: float,
    pixel_size: float,
    magnification: float,
) -> np.ndarray:
    """Convert the physical point spread function (PSF) parameter to the dimensionless 'PhotonSig'.

    Args:
        psf_xy (np.ndarray): PSF as a physical parameter
        emission_wavelength (float): Emission wavelength in nm.
        numerical_aperture (float): Numerical aperture of the image acquistion setup.
        pixel_size (float): The pixel size in um.
        magnification (float): The objective magnification used during image acquisition.

    Returns:
        np.ndarray: 'PhotonSig' as a dimensionless parameter.
    """
    # Assume that the pixel size is provided in micro metres
    pixel_size = pixel_size * 1e-6
    # and that the emission wavelength is provided in nanometres
    emission_wavelength = emission_wavelength * 1e-9

    pixel_size = pixel_size / magnification

    return ((psf_xy * emission_wavelength) / (2 * numerical_aperture)) / pixel_size


# %%
if __name__ == "__main__":
    # Setting tolerence for the tests.
    # These are necessary as the floating point values reported in the paper are truncated.
    # This might reduce the 'power' of the equivalence tests though.
    ATOL = 1e-3
    RTOL = 1e-3

    # Testing based on the descriptions of the paper.
    # Based on the paper's supplementary materials.
    minval = 0.1082
    maxval = 2.1634
    𝑚𝑜𝑑 = 0.4838
    𝑚𝑜𝑑𝑆𝑖𝑔 = 4.4715

    assert np.allclose(
        get_minmax_from_modsig(𝑚𝑜𝑑, 𝑚𝑜𝑑𝑆𝑖𝑔, scale="log"),
        (minval, maxval),
        rtol=RTOL,
        atol=ATOL,
    ), f'get_minmax_from_modsig(𝑚𝑜𝑑, 𝑚𝑜𝑑𝑆𝑖𝑔, scale="log") should return ({minval}, {maxval}). Got {get_minmax_from_modsig(𝑚𝑜𝑑, 𝑚𝑜𝑑𝑆𝑖𝑔, scale="log")}'
    assert np.allclose(
        get_modsig_from_minmax(minval, maxval, scale="log"),
        (𝑚𝑜𝑑, 𝑚𝑜𝑑𝑆𝑖𝑔),
        rtol=RTOL,
        atol=ATOL,
    ), f'get_modsig_from_minmax(minval, maxval, scale="log") should return ({𝑚𝑜𝑑}, {𝑚𝑜𝑑𝑆𝑖𝑔}). Got {get_modsig_from_minmax(minval, maxval, scale="log")}'

    # also test if these functions are reversible.
    _mod, _modsig = get_modsig_from_minmax(minval, maxval, scale="log")
    assert np.allclose(
        get_minmax_from_modsig(_mod, _modsig, scale="log"),
        (minval, maxval),
        rtol=RTOL,
        atol=ATOL,
    ), f'get_minmax_from_modsig(_mod, _modsig, scale="log") should return ({minval}, {maxval}). Got ({_mod}, {_modsig})'

    # For these tests, we assume the min/max values provided in supplementary table 1.4 must be close to those in supplementary table 1.5
    PARTICLE_SIG_D_TEST_CONDITIONS_DICT = {
        "fcsnet_2d_setting_1": {
            "d_min": 0.02,
            "d_max": 15.0,
            "particle_sig_min": 0.0086,
            "particle_sig_max": 0.2350,
            "frame_time": 0.00106,
            "steps_per_frame": 10,
            "pixel_size": 24,
            "magnification": 100,
        },
        "imfcsnet_2d_setting_1": {
            "d_min": 0.02,
            "d_max": 50.0,
            "particle_sig_min": 0.0086,
            "particle_sig_max": 0.4290,
            "frame_time": 0.00106,
            "steps_per_frame": 10,
            "pixel_size": 24,
            "magnification": 100,
        },
        "imfcsnet_2d_setting_2": {
            "d_min": 0.02,
            "d_max": 50.0,
            "particle_sig_min": 0.0120,
            "particle_sig_max": 0.5980,
            "frame_time": 0.00206,
            "steps_per_frame": 10,
            "pixel_size": 24,
            "magnification": 100,
        },
        "imfcsnet_3d": {
            "d_min": 0.02,
            "d_max": 50.0,
            "particle_sig_min": 0.0071,
            "particle_sig_max": 0.3571,
            "frame_time": 0.00204,
            "steps_per_frame": 10,
            "pixel_size": 24,
            "magnification": 60,
        },
    }

    for setting_name, d in PARTICLE_SIG_D_TEST_CONDITIONS_DICT.items():
        ## Testing D to PARTICLE_SIG
        # Testing min
        _val = convert_particle_sig_to_d(
            particle_sig=d["particle_sig_min"],
            frame_time=d["frame_time"],
            steps_per_frame=d["steps_per_frame"],
            pixel_size=d["pixel_size"],
            magnification=d["magnification"],
        )
        assert np.allclose(
            _val,
            d["d_min"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting PARTICLE_SIG {d["particle_sig_min"]} to D {d["d_min"]}. Got {_val}'
        # Testing max
        _val = convert_particle_sig_to_d(
            particle_sig=d["particle_sig_max"],
            frame_time=d["frame_time"],
            steps_per_frame=d["steps_per_frame"],
            pixel_size=d["pixel_size"],
            magnification=d["magnification"],
        )
        assert np.allclose(
            _val,
            d["d_max"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting PARTICLE_SIG {d["particle_sig_max"]} to D {d["d_max"]}. Got {_val}'

        ## Testing PARTICLE_SIG to D
        # Testing min
        _val = convert_d_to_particle_sig(
            d=d["d_min"],
            frame_time=d["frame_time"],
            steps_per_frame=d["steps_per_frame"],
            pixel_size=d["pixel_size"],
            magnification=d["magnification"],
        )
        assert np.allclose(
            _val,
            d["particle_sig_min"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting D {d["d_min"]} to PARTICLE_SIG {d["particle_sig_min"]}. Got {_val}'
        # Testing max
        _val = convert_d_to_particle_sig(
            d=d["d_max"],
            frame_time=d["frame_time"],
            steps_per_frame=d["steps_per_frame"],
            pixel_size=d["pixel_size"],
            magnification=d["magnification"],
        )
        assert np.allclose(
            _val,
            d["particle_sig_max"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting D {d["d_max"]} to PARTICLE_SIG {d["particle_sig_max"]}. Got {_val}'

    PARTICLE_DENSITY_N_TEST_CONDITIONS_DICT = {
        "fcsnet_2d": {
            "n_min": 358,
            "n_max": 2389,
            "particle_density_min": 0.1623,
            "particle_density_max": 1.0817,
            "pixels_per_side": 47,
        },
        "imfcsnet_2d": {
            "n_min": 24,
            "n_max": 486,
            "particle_density_min": 0.1082,
            "particle_density_max": 2.1634,
            "pixels_per_side": 15,
        },
    }

    for setting_name, d in PARTICLE_DENSITY_N_TEST_CONDITIONS_DICT.items():
        ## Testing PARTICLE_DENSITY to N
        # Testing min
        _val = convert_particle_density_to_n(
            particle_density=d["particle_density_min"],
            pixels_per_side=d["pixels_per_side"],
        )
        assert np.allclose(
            _val,
            d["n_min"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting PARTICLE_DENSITY {d["particle_density_min"]} to N {d["n_min"]}. Got {_val}'
        # Testing max
        _val = convert_particle_density_to_n(
            particle_density=d["particle_density_max"],
            pixels_per_side=d["pixels_per_side"],
        )
        assert np.allclose(
            _val,
            d["n_max"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting PARTICLE_DENSITY {d["particle_density_max"]} to N {d["n_max"]}. Got {_val}'

        ## Testing N to PARTICLE_DENSITY
        # Note that here, we increase the tolerence, as N can only exist as ints, so the inverse conversion can be sensitive to the 'quantization' effect.
        # Testing min
        _val = convert_n_to_particle_density(
            n=d["n_min"],
            pixels_per_side=d["pixels_per_side"],
        )
        assert np.allclose(
            _val,
            d["particle_density_min"],
            rtol=RTOL * 1e1,
            atol=ATOL * 1e1,
        ), f'Failed test for case {setting_name} when converting N {d["n_min"]} to PARTICLE_DENSITY {d["particle_density_min"]}. Got {_val}'
        # Testing max
        _val = convert_n_to_particle_density(
            n=d["n_max"],
            pixels_per_side=d["pixels_per_side"],
        )
        assert np.allclose(
            _val,
            d["particle_density_max"],
            rtol=RTOL * 1e1,
            atol=ATOL * 1e1,
        ), f'Failed test for case {setting_name} when converting N {d["n_max"]} to PARTICLE_DENSITY {d["particle_density_max"]}. Got {_val}'

    EMISSION_RATE_CPS_TEST_CONDITIONS_DICT = {
        "fcsnet_2d": {
            "cps_min": 4000,
            "cps_max": 9000,
            "emission_rate_min": 0.424,
            "emission_rate_max": 0.954,
            "frame_time": 0.00106,
            "steps_per_frame": 10,
        },
        "imfcsnet_2d": {
            "cps_min": 1000,
            "cps_max": 10000,
            "emission_rate_min": 0.106,
            "emission_rate_max": 1.060,
            "frame_time": 0.00106,
            "steps_per_frame": 10,
        },
        "imfcsnet_3d": {
            "cps_min": 1000,
            "cps_max": 10000,
            "emission_rate_min": 0.204,
            "emission_rate_max": 2.04,
            "frame_time": 0.00204,
            "steps_per_frame": 10,
        },
    }

    for setting_name, d in EMISSION_RATE_CPS_TEST_CONDITIONS_DICT.items():
        ## Testing EMISSION_RATE to CPS
        # Testing min
        _val = convert_emission_rate_to_cps(
            emission_rate=d["emission_rate_min"],
            frame_time=d["frame_time"],
            steps_per_frame=d["steps_per_frame"],
        )
        assert np.allclose(
            _val,
            d["cps_min"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting EMISSION_RATE {d["emission_rate_min"]} to CPS {d["cps_min"]}. Got {_val}'
        # Testing max
        _val = convert_emission_rate_to_cps(
            emission_rate=d["emission_rate_max"],
            frame_time=d["frame_time"],
            steps_per_frame=d["steps_per_frame"],
        )
        assert np.allclose(
            _val,
            d["cps_max"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting EMISSION_RATE {d["emission_rate_max"]} to CPS {d["cps_max"]}. Got {_val}'

        ## Testing CPS to EMISSION_RATE
        _val = convert_cps_to_emission_rate(
            cps=d["cps_min"],
            frame_time=d["frame_time"],
            steps_per_frame=d["steps_per_frame"],
        )
        assert np.allclose(
            _val,
            d["emission_rate_min"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting CPS {d["cps_min"]} to EMISSION_RATE {d["emission_rate_min"]}. Got {_val}'
        # Testing max
        _val = convert_cps_to_emission_rate(
            cps=d["cps_max"],
            frame_time=d["frame_time"],
            steps_per_frame=d["steps_per_frame"],
        )
        assert np.allclose(
            _val,
            d["emission_rate_max"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting CPS {d["cps_max"]} to EMISSION_RATE {d["emission_rate_max"]}. Got {_val}'

    PHOTON_SIG_PSF_TEST_CONDITIONS_DICT = {
        "fcsnet_2d_setting_1": {
            "psf_xy_min": 0.75,
            "psf_xy_max": 0.85,
            "photon_sig_min": 0.6282,
            "photon_sig_max": 0.7120,
            "emission_wavelength": 583,
            "numerical_aperture": 1.45,
            "pixel_size": 24,
            "magnification": 100,
        },
        "imfcsnet_2d_setting_1": {
            "psf_xy_min": 0.75,
            "psf_xy_max": 0.85,
            "photon_sig_min": 0.6282,
            "photon_sig_max": 0.7120,
            "emission_wavelength": 583,
            "numerical_aperture": 1.45,
            "pixel_size": 24,
            "magnification": 100,
        },
        "imfcsnet_2d_setting_2": {
            "psf_xy_min": 0.96,
            "psf_xy_max": 1.06,
            "photon_sig_min": 0.6805,
            "photon_sig_max": 0.7514,
            "emission_wavelength": 507,
            "numerical_aperture": 1.49,
            "pixel_size": 24,
            "magnification": 100,
        },
        "imfcsnet_3d": {
            "psf_xy_min": 1.05,
            "psf_xy_max": 1.15,
            "photon_sig_min": 0.6759,
            "photon_sig_max": 0.7403,
            "emission_wavelength": 515,
            "numerical_aperture": 1.0,
            "pixel_size": 24,
            "magnification": 60,
        },
    }

    for setting_name, d in PHOTON_SIG_PSF_TEST_CONDITIONS_DICT.items():
        ## Testing PSF to PHOTON_SIG
        # Testing min
        _val = convert_photon_sig_to_psf(
            photon_sig=d["photon_sig_min"],
            emission_wavelength=d["emission_wavelength"],
            numerical_aperture=d["numerical_aperture"],
            pixel_size=d["pixel_size"],
            magnification=d["magnification"],
        )
        assert np.allclose(
            _val,
            d["psf_xy_min"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting PHOTON_SIG {d["photon_sig_min"]} to PSF_XY {d["psf_xy_min"]}. Got {_val}'
        # Testing max
        _val = convert_photon_sig_to_psf(
            photon_sig=d["photon_sig_max"],
            emission_wavelength=d["emission_wavelength"],
            numerical_aperture=d["numerical_aperture"],
            pixel_size=d["pixel_size"],
            magnification=d["magnification"],
        )
        assert np.allclose(
            _val,
            d["psf_xy_max"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting PHOTON_SIG {d["photon_sig_max"]} to PSF_XY {d["psf_xy_max"]}. Got {_val}'

        ## Testing PHOTON_SIG to PSF
        # Testing min
        _val = convert_psf_to_photon_sig(
            psf_xy=d["psf_xy_min"],
            emission_wavelength=d["emission_wavelength"],
            numerical_aperture=d["numerical_aperture"],
            pixel_size=d["pixel_size"],
            magnification=d["magnification"],
        )
        assert np.allclose(
            _val,
            d["photon_sig_min"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting PSF_XY {d["psf_xy_min"]} to PHOTON_SIG {d["photon_sig_min"]}. Got {_val}'
        # Testing max
        _val = convert_psf_to_photon_sig(
            psf_xy=d["psf_xy_max"],
            emission_wavelength=d["emission_wavelength"],
            numerical_aperture=d["numerical_aperture"],
            pixel_size=d["pixel_size"],
            magnification=d["magnification"],
        )
        assert np.allclose(
            _val,
            d["photon_sig_max"],
            rtol=RTOL,
            atol=ATOL,
        ), f'Failed test for case {setting_name} when converting PSF_XY {d["psf_xy_max"]} to PHOTON_SIG {d["photon_sig_max"]}. Got {_val}'
