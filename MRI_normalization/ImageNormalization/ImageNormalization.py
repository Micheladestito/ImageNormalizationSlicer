import logging
import os
from typing import Annotated, Optional
import argparse
import typing
import slicer
from scipy.signal import argrelmax
import statsmodels.api as sm
import argparse
import collections.abc
import typing
import numpy as np
from scipy.signal import argrelmax
import statsmodels.api as sm
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
import logging
import nibabel as nib

logger = logging.getLogger(__name__)

# If needed install libraries before imporing. If already installed, just import it.
try:
  import intensity_normalization 

except ModuleNotFoundError:
  slicer.util.pip_install("intensity_normalization")
  import intensity_normalization.normalize.base as intnormb
  import intensity_normalization.typing as intnormt
  import intensity_normalization.util.histogram_tools as intnormhisttool
  import intensity_normalization.util.histogram_tools as intnormhisttool



from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode

import numpy as np
import SimpleITK as sitk
import sitkUtils
import numpy.typing as npt

#
# ImageNormalization
#


class ImageNormalization(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("ImageNormalization")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = ["Filtering"]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Michela Destito (Magna Graecia University of Catanzaro (Italy))", "Paolo Zaffino (Magna Graecia University of Catanzaro (Italy))", "Maria Francesca Spadea (Karlsruher Intitute of Technology (Germany))"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#ImageNormalization">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # ImageNormalization1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="ImageNormalization",
        sampleName="ImageNormalization1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "ImageNormalization1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="ImageNormalization1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="ImageNormalization1",
    )

    # ImageNormalization2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="ImageNormalization",
        sampleName="ImageNormalization2",
        thumbnailFileName=os.path.join(iconsPath, "ImageNormalization2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="ImageNormalization2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="ImageNormalization2",
    )


#
# ImageNormalizationParameterNode
#


@parameterNodeWrapper
class ImageNormalizationParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# ImageNormalizationWidget
#


class ImageNormalizationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        
    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/ImageNormalization.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ImageNormalizationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        
        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.zscoreButton.connect('toggled(bool)', self.onZScoreButton)
        self.ui.whiteStripeButton.connect('toggled(bool)', self.onWhiteStripeButton)
        self.ui.nyulButton.connect('toggled(bool)', self.onNyulButton)


        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        #####self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        #if not self._parameterNode.inputVolume:
        #    firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #    if firstVolumeNode:
        #        self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[ImageNormalizationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            #self._checkCanApply()
   

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False


    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        if self.ui.zscoreButton.isChecked():
            normalizationMethod = 'zscore'
        elif self.ui.whiteStripeButton.isChecked():
            normalizationMethod = 'whiteStripe'
        elif self.ui.nyulButton.isChecked():
            normalizationMethod = 'Nyul'    
        print(normalizationMethod)
        self.logic.runNormalization(self.ui.inputSelector.currentNode().GetName(),
                                    self.ui.outputSelector.currentNode().GetName(),
                                    self.ui.DirectoryButton.directory,
                                    normalizationMethod)
    
    def onZScoreButton(self, toggle):
        self.ui.DirectoryButton.setEnabled(False)
        self.ui.inputSelector.setEnabled(True)

    def onWhiteStripeButton(self, toggle):
        self.ui.DirectoryButton.setEnabled(False)
        self.ui.inputSelector.setEnabled(True)

    def onNyulButton(self, toggle):
        self.ui.DirectoryButton.setEnabled(True)
        self.ui.inputSelector.setEnabled(False)

   
#
# ImageNormalizationLogic
#

def zscore_normalize(img):
    mask_data = img > img.mean()
    logical_mask = mask_data == 1  # force the mask to be logical type
    mean = img[logical_mask].mean()
    std = img[logical_mask].std()
    normalized = ((img - mean) / std)
    return normalized



def smooth_hist(data):
    """
    use KDE to get smooth estimate of histogram

    Args:
        data (np.ndarray): array of image data

    Returns:
        grid (np.ndarray): domain of the pdf
        pdf (np.ndarray): kernel density estimate of the pdf of data
    """
    data = data.flatten().astype(np.float64)
    bw = data.max() / 80

    kde = sm.nonparametric.KDEUnivariate(data)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    pdf = 100.0 * kde.density
    grid = kde.support

    return grid, pdf


def get_largest_mode(data):
    """
    gets the last (reliable) peak in the histogram

    Args:
        data (np.ndarray): image data

    Returns:
        largest_peak (int): index of the largest peak
    """
    grid, pdf = smooth_hist(data)
    largest_peak = grid[np.argmax(pdf)]
    return largest_peak


def get_last_mode(data, rare_prop=96, remove_tail=True):
    """
    gets the last (reliable) peak in the histogram

    Args:
        data (np.ndarray): image data
        rare_prop (float): if remove_tail, use the proportion of hist above
        remove_tail (bool): remove rare portions of histogram
            (included to replicate the default behavior in the R version)

    Returns:
        last_peak (int): index of the last peak
    """
    if remove_tail:
        rare_thresh = np.percentile(data, rare_prop)
        which_rare = data >= rare_thresh
        data = data[which_rare != 1]
    grid, pdf = smooth_hist(data)
    maxima = argrelmax(pdf)[0]  # for some reason argrelmax returns a tuple, so [0] extracts value
    last_peak = grid[maxima[-1]]
    return last_peak


def get_first_mode(data, rare_prop=99, remove_tail=True):
    """
    gets the first (reliable) peak in the histogram

    Args:
        data (np.ndarray): image data
        rare_prop (float): if remove_tail, use the proportion of hist above
        remove_tail (bool): remove rare portions of histogram
            (included to replicate the default behavior in the R version)

    Returns:
        first_peak (int): index of the first peak
    """
    if remove_tail:
        rare_thresh = np.percentile(data, rare_prop)
        which_rare = data >= rare_thresh
        data = data[which_rare != 1]
    grid, pdf = smooth_hist(data)
    maxima = argrelmax(pdf)[0]  # for some reason argrelmax returns a tuple, so [0] extracts value
    first_peak = grid[maxima[0]]
    return first_peak


def whitestripe(img, contrast, mask=None, width=0.05, width_l=None, width_u=None):
    """
    find the "(normal appearing) white (matter) stripe" of the input MR image
    and return the indices

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        contrast (str): contrast of img (e.g., T1)
        mask (nibabel.nifti1.Nifti1Image): brainmask for img (None is default, for skull-stripped img)
        width (float): width quantile for the "white (matter) stripe"
        width_l (float): lower bound for width (default None, derives from width)
        width_u (float): upper bound for width (default None, derives from width)

    Returns:
        ws_ind (np.ndarray): the white stripe indices (boolean mask)
    """
    if width_l is None and width_u is None:
        width_l = width
        width_u = width
    if mask is not None:
        mask_data = mask.get_data()
        masked = img * mask_data
        voi = img[mask_data == 1]
    else:
        masked = img
        voi = img[img > img.mean()]
    if contrast.lower() in ['t1', 'last']:
        mode = get_last_mode(voi)
    elif contrast.lower() in ['t2', 'flair', 'largest']:
        mode = get_last_mode(voi)
    elif contrast.lower() in ['md', 'first']:
        mode = get_last_mode(voi)
    else:
        raise NormalizationError('Contrast {} not valid, needs to be `t1`,`t2`,`flair`,`md`,`first`,`largest`,`last`'.format(contrast))
    img_mode_q = np.mean(voi < mode)
    ws = np.percentile(voi, (max(img_mode_q - width_l, 0) * 100, min(img_mode_q + width_u, 1) * 100))
    ws_ind = np.logical_and(masked > ws[0], masked < ws[1])
    if len(ws_ind) == 0:
        raise NormalizationError('WhiteStripe failed to find any valid indices!')
    return ws_ind


def whitestripe_norm(img, indices):
    """
    use the whitestripe indices to standardize the data (i.e., subtract the
    mean of the values in the indices and divide by the std of those values)

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        indices (np.ndarray): whitestripe indices (see whitestripe func)

    Returns:
        norm_img (nibabel.nifti1.Nifti1Image): normalized image in nifti format
    """
    
    mu = np.mean(img[indices])
    sig = np.std(img[indices])
    norm_img_data = (img - mu)/sig
    return norm_img_data



def split_filename(filepath):
    """ split a filepath into the full path, filename, and extension (works with .nii.gz) """
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def open_nii(filepath):
    """ open a nifti file with nibabel and return the object """
    image = os.path.abspath(os.path.expanduser(filepath))
    obj = nib.load(image)
    return obj


def save_nii(obj, outfile, data=None, is_nii=False):
    """ save a nifti object """
    if not is_nii:
        if data is None:
            data = obj.get_data()
        nib.Nifti1Image(data, obj.affine, obj.header)\
            .to_filename(outfile)
    else:
        obj.to_filename(outfile)


def glob_nii(dir):
    """ return a sorted list of nifti files for a given directory """
    fns = sorted(glob(os.path.join(dir, '*.nii*')))
    return fns

def nyul_normalize(img_dir, mask_dir=None, output_dir=None, standard_hist=None, write_to_disk=True):
    
    input_files = glob_nii(img_dir)
    if output_dir is None:
        out_fns = [None] * len(input_files)
    else:
        out_fns = []
        for fn in input_files:
            _, base, ext = split_filename(fn)
            out_fns.append(os.path.join(output_dir, base + '_hm' + ext))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    mask_files = [None] * len(input_files) if mask_dir is None else io.glob_nii(mask_dir)

    if standard_hist is None:
        logger.info('Learning standard scale for the set of images')
        standard_scale, percs = train(input_files, mask_files)
    elif not os.path.isfile(standard_hist):
        logger.info('Learning standard scale for the set of images')
        standard_scale, percs = train(input_files, mask_files)
        np.save(standard_hist, np.vstack((standard_scale, percs)))
    else:
        logger.info('Loading standard scale ({}) for the set of images'.format(standard_hist))
        standard_scale, percs = np.load(standard_hist)

    normalized = None
    for i, (img_fn, mask_fn, out_fn) in enumerate(zip(input_files, mask_files, out_fns)):
        _, base, _ = io.split_filename(img_fn)
        logger.info('Transforming image {} to standard scale ({:d}/{:d})'.format(base, i+1, len(input_files)))
        img = open_nii(img_fn)
        mask = open_nii(mask_fn) if mask_fn is not None else None
        normalized = do_hist_norm(img, percs, standard_scale, mask)
        if write_to_disk:
            save_nii(normalized, out_fn, is_nii=True)

    return normalized


def get_landmarks(img, percs):
    """
    get the landmarks for the Nyul and Udupa norm method for a specific image

    Args:
        img (np.ndarray): image on which to find landmarks
        percs (np.ndarray): corresponding landmark percentiles to extract

    Returns:
        landmarks (np.ndarray): intensity values corresponding to percs in img
    """
    landmarks = np.percentile(img, percs)
    return landmarks


def train(img_fns, mask_fns=None, i_min=1, i_max=99, i_s_min=1, i_s_max=100, l_percentile=10, u_percentile=90, step=10):
    """
    determine the standard scale for the set of images

    Args:
        img_fns (list): set of NifTI MR image paths which are to be normalized
        mask_fns (list): set of corresponding masks (if not provided, estimated)
        i_min (float): minimum percentile to consider in the images
        i_max (float): maximum percentile to consider in the images
        i_s_min (float): minimum percentile on the standard scale
        i_s_max (float): maximum percentile on the standard scale
        l_percentile (int): middle percentile lower bound (e.g., for deciles 10)
        u_percentile (int): middle percentile upper bound (e.g., for deciles 90)
        step (int): step for middle percentiles (e.g., for deciles 10)

    Returns:
        standard_scale (np.ndarray): average landmark intensity for images
        percs (np.ndarray): array of all percentiles used
    """
    mask_fns = [None] * len(img_fns) if mask_fns is None else mask_fns
    percs = np.concatenate(([i_min], np.arange(l_percentile, u_percentile+1, step), [i_max]))
    standard_scale = np.zeros(len(percs))
    for i, (img_fn, mask_fn) in enumerate(zip(img_fns, mask_fns)):
        img_data = open_nii(img_fn).get_data()
        mask = open_nii(mask_fn) if mask_fn is not None else None
        mask_data = img_data > img_data.mean() if mask is None else mask.get_data()
        masked = img_data[mask_data > 0]
        landmarks = get_landmarks(masked, percs)
        min_p = np.percentile(masked, i_min)
        max_p = np.percentile(masked, i_max)
        f = interp1d([min_p, max_p], [i_s_min, i_s_max])
        landmarks = np.array(f(landmarks))
        standard_scale += landmarks
    standard_scale = standard_scale / len(img_fns)
    return standard_scale, percs


def do_hist_norm(img, landmark_percs, standard_scale, mask=None):
    """
    do the Nyul and Udupa histogram normalization routine with a given set of learned landmarks

    Args:
        img (nibabel.nifti1.Nifti1Image): image on which to find landmarks
        landmark_percs (np.ndarray): corresponding landmark points of standard scale
        standard_scale (np.ndarray): landmarks on the standard scale
        mask (nibabel.nifti1.Nifti1Image): foreground mask for img

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): normalized image
    """
    mask_data = img > img.mean() if mask is None else mask.get_data()
    masked = img[mask_data > 0]
    landmarks = get_landmarks(masked, landmark_percs)
    f = interp1d(landmarks, standard_scale, fill_value='extrapolate')
    normed = f(img)
    return normed


class ImageNormalizationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def runNormalization(self, inputNodeName, outputNodeName,directory,normalizationMethod):
        inputVolumeSitk = sitk.Cast(sitkUtils.PullVolumeFromSlicer(inputNodeName), sitk.sitkFloat32)
        inputVolume = sitk.GetArrayFromImage(inputVolumeSitk).astype(np.float32)
        if normalizationMethod == 'zscore':
            inputVolume = zscore_normalize(inputVolume)
        elif normalizationMethod == 'whiteStripe':
            indices=whitestripe(inputVolume, contrast='T1', mask=None, width=0.05, width_l=None, width_u=None)
            inputVolume= whitestripe_norm(inputVolume, indices)
        elif normalizationMethod == 'Nyul':
            directory_=os.listdir(directory)
            image_path=[]
            for image in directory_:
                print(directory)
                path=os.path.join(directory, image)
                image_path.append(path)
            images = [sitk.ReadImage(image_path) for image_path in image_path]
            landmark_percs, standard_scale=train(image_path, mask_fns=None, i_min=1, i_max=99, i_s_min=1, i_s_max=100, l_percentile=10, u_percentile=90, step=10)
            inputVolume=do_hist_norm(inputVolume,landmark_percs, standard_scale, mask=None)
            
            return
        
        normalizedVolumeSitk = sitk.GetImageFromArray(inputVolume)
        normalizedVolumeSitk.CopyInformation(inputVolumeSitk)

        sitkUtils.PushVolumeToSlicer(normalizedVolumeSitk, outputNodeName)
        


#
# ImageNormalizationTest
#


class ImageNormalizationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_ImageNormalization1()

    def test_ImageNormalization1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("ImageNormalization1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    

        # Test the module logic

        logic = ImageNormalizationLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
