# -------------------------------------------------------------------------------
# CONSTANTS
#
# Author: Maleakhi A. Wijaya
# Description: This file contains constants, including shift intensity levels,
#     shift type, dimensionality reduction method, etc.
# -------------------------------------------------------------------------------

from enum import Enum

# -------------------------------------------------------------------------------
## Constants

MAJORITY = "MAJORITY"
DSPRITES_CONCEPT_NAMES = ["color", "shape", "scale", "rotation", "x", "y"]
SMALLNORB_CONCEPT_NAMES = ['category', 'instance', 'elevation', 'azimuth', 'lighting']
SHAPES3D_CONCEPT_NAMES = ["floor", "wall", "object", "scale", "shape", "orientation"]


# -------------------------------------------------------------------------------
## Shift configurations

class ShiftIntensity(Enum):
    """
    Constants for shift intensity to be applied.
    """

    Small = 0
    Medium = 1
    Large = 2


class ShiftType(Enum):
    """
    Constants for shift types to be applied
    """

    # Image shifts
    Width = 0
    Height = 1
    Rotation = 2
    Shear = 3
    Zoom = 4
    Flip = 5
    All = 6

    # Gaussian shift
    Gaussian = 7

    # Knockout shift
    Knockout = 8

    # Concept shift
    Concept = 9

    # Adversarial shift
    Adversarial = 10


class ImageDataGeneratorConfig:
    """
    Configuration for ImageDataGeneratior.
    """

    # Rotation
    Rotation = {
        ShiftIntensity.Small: 10,
        ShiftIntensity.Medium: 40,
        ShiftIntensity.Large: 90
    }

    # Width shift (x-translation)
    Width = {
        ShiftIntensity.Small: 0.05,
        ShiftIntensity.Medium: 0.2,
        ShiftIntensity.Large: 0.4
    }

    # Height shift (y-translation)
    Height = {
        ShiftIntensity.Small: 0.05,
        ShiftIntensity.Medium: 0.2,
        ShiftIntensity.Large: 0.4
    }

    # Shear
    Shear = {
        ShiftIntensity.Small: 0.1,
        ShiftIntensity.Medium: 0.2,
        ShiftIntensity.Large: 0.3
    }

    # Zoom
    Zoom = {
        ShiftIntensity.Small: 0.1,
        ShiftIntensity.Medium: 0.2,
        ShiftIntensity.Large: 0.4
    }

    # Flip
    Flip = {
        ShiftIntensity.Small: (False, False),
        ShiftIntensity.Medium: (True, False),
        ShiftIntensity.Large: (True, True)
    }


# -------------------------------------------------------------------------------
## Dimensionality reductor configuration

class DimensionalityReductor(Enum):
    BBSDs = 0
    BBSDh = 1
    CBSDs = 2
    CBSDh = 3
    PCA = 4
    SRP = 5
    UAE = 6
    TAE = 7


class ConceptBottleneckTraining(Enum):
    """
    See the concept bottleneck models paper for the difference between these
    training approaches.
    """
    Independent = 0
    Sequential = 1
    Joint = 2


# -------------------------------------------------------------------------------
## Statistical tests configuration

class OneDimensionalTest(Enum):
    KS = 0
    AD = 1


# -------------------------------------------------------------------------------
## Dataset configuration

class Dataset(Enum):
    """
    Constants representing the dataset.
    """

    DSPRITES = 0
    SHAPES3D = 1
    SMALLNORB = 2


class DatasetTask(Enum):
    """
    Constants representing dataset tasks.
    Task1: predict one of the concepts as the end task.
    Task2: predict two combinations of concepts as the end task.
    Task3: predict binary category of a concept.
    """

    Task1 = 0
    Task2 = 1
    Task3 = 2


# -------------------------------------------------------------------------------
## Classifier two sample test

class ClassifierTwoSampleTest(Enum):
    """
    Various model options for the classifier two sample tests.
    FFNN: the end-to-end feedforward neural network model.
    CBM: the concept bottleneck option.
    CME: the concept model extraction model of the end-to-end FFNN.
    OCSVM: the one class SVM (popular models for novelty detection).
    LDA: linear discriminant analysis (fails gracefully when assumption about distribution is violated).
    """

    FFNN = 0
    CBM = 1
    OCSVM = 2  # note that the one-class SVM is used only for computing most likely shifted samples.
    LDA = 3