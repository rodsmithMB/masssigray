"""massSigray: sample planning for lab-based XAS (pellets, foils, fluorescence)."""
from .detector import DetectorModel, calibrate, Scan
from .flux import FluxSpectrum, load_flux
from .materials import MATRIX_PRESETS, Material, matrix_preset
from .scan import ScanPlan
from .transmission import (PelletConstraint, TransmissionSetup, evaluate_masses, foil_setup,
                           optimize)

__version__ = "0.1.0"
