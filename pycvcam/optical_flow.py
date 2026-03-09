from typing import Any, Dict, Optional, Tuple
from numbers import Integral, Real
from numpy.typing import ArrayLike

import numpy
import matplotlib.pyplot as plt
import cv2


def compute_optical_flow(
    image1: ArrayLike,
    image2: ArrayLike,
    channel: Integral = 0,
    estimate_flow_x: Optional[ArrayLike] = None,
    estimate_flow_y: Optional[ArrayLike] = None,
    region: Optional[Tuple[Integral, Integral, Integral, Integral]] = None,
    disflow_params: Optional[Dict[str, Any]] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Compute the optical flow between two images using the DIS method of ``OpenCV``.

    The optical flow is computed between two images using the Dense Inverse Search (DIS)
    algorithm implemented in OpenCV. The function allows selecting a specific channel
    from multi-channel images and provides options to customize the DIS algorithm
    through a parameters dictionary.

    Lets consider two images :math:`I_1` and :math:`I_2` of shape :math:`(H, W)`. The
    optical flow is represented as two 2D arrays :math:`F_x` and :math:`F_y`, each of
    shape :math:`(H, W)`, where :math:`F_x[i, j]` and :math:`F_y[i, j]` denote the
    horizontal and vertical displacements (in pixels) of the pixel located at
    :math:`(i, j)` in the first image to its corresponding position in the second image.

    .. math::

        I_2(i + F_y[i, j], j + F_x[i, j]) \approx I_1(i, j)

    With image array indices :math:`i` and :math:`j` representing the row (vertical)
    and column (horizontal) indices, respectively.

    .. note::

        - The method convert the images data to `uint8` before processing, as required by OpenCV's DIS optical flow implementation.
        - The method convert the estimated flow to `float32`, as required by OpenCV's DIS optical flow implementation.

    The ``disflow_params`` dictionary can contain the following keys:

    - ``"FinestScale"``: :class:`int` (default: ``0``) - The finest scale to use in the image pyramid.
    - ``"VariationalRefinementAlpha"``: :class:`float` (default: ``1``) - Weight for the smoothness term in the variational refinement.
    - ``"VariationalRefinementDelta"``: :class:`float` (default: ``1``) - Weight for the data term in the variational refinement.
    - ``"VariationalRefinementGamma"``: :class:`float` (default: ``1``) - Weight for the gradient constancy term in the variational refinement.
    - ``"VariationalRefinementEpsilon"``: :class:`float` (default: ``0.02``) -- Small constant to avoid division by zero in the variational refinement.
    - ``"VariationalRefinementIterations"``: :class:`int` (default: ``10``) - Number of iterations for the variational refinement.
    - ``"UseMeanNormalization"``: :class:`bool` (default: ``True``) - Whether to use mean normalization.
    - ``"GradientDescentIterations"``: :class:`int` (default: ``500``) - Number of gradient descent iterations.
    - ``"UseSpatialPropagation"``: :class:`bool` (default: ``True``) - Whether to use spatial propagation.
    - ``"PatchSize"``: :class:`int` (default: ``50``) - Size of the patches.
    - ``"PatchStride"``: :class:`int` (default: ``10``) - Stride between patches.

    .. warning::

        ``x`` and ``y`` components of the optical flow are defined in image coordinates
        and not in array indices. This means that the x-component corresponds to
        the horizontal displacement (columns) and the y-component corresponds to the
        vertical displacement (rows).

        The optical flow is compute with :obj:`numpy.uint8` images, so the input images
        will be scaled to fit into the range of :obj:`numpy.uint8` if they are of a
        different unsigned integer type. The output flow will be in pixels and can be
        positive or negative depending on the direction of motion.


    Parameters
    ----------
    image1: ArrayLike
        The first image with shape :math:`(H, W)` or :math:`(H, W, C)` where
        :math:`C` is the number of channels.
        The image must be unsigned integer type.

    image2: ArrayLike
        The second image with shape :math:`(H, W)` or :math:`(H, W, C)` where
        :math:`C` is the number of channels.
        The image must be unsigned integer type.

    channel: Integral, optional
        The channel of the images to use for optical flow computation. Default is 0.

    estimate_flow_x : Optional[ArrayLike], optional
        An initial estimate for the x-component of the optical flow with shape
        :math:`(H, W)`. Default is None.

    estimate_flow_y : Optional[ArrayLike], optional
        An initial estimate for the y-component of the optical flow with shape
        :math:`(H, W)`. Default is None.

    region: Optional[Tuple[Integral, Integral, Integral, Integral]], optional
        A tuple specifying the region of interest in the format (x, y, width, height).
        If None, the entire image is computed. Default is None.

    disflow_params: Optional[Dict[str, Any]], optional
        Parameters for the DIS optical flow algorithm. See above for details.
        Default is None.


    Returns
    -------
    flow_x: :class:`numpy.ndarray`
        The x-component of the optical flow (horizontal displacement) in pixels with
        shape :math:`(H, W)`.
        The pixels outside the specified region (if any) will be set to :obj:`numpy.nan`.

    flow_y: :class:`numpy.ndarray`
        The y-component of the optical flow (vertical displacement) in pixels with
        shape :math:`(H, W)`.
        The pixels outside the specified region (if any) will be set to :obj:`numpy.nan`.


    See Also
    --------
    pycvcam.display_optical_flow
        Display the optical flow overlaid on the given image using Matplotlib.


    Examples
    --------
    Create two example images and compute the optical flow between them.

    .. figure:: /_static/textures/lena_texture.png
        :align: center
        :width: 50%

        Lena image used for the example.

    .. code-block:: python
        :linenos:

        import numpy
        from pycvcam import compute_optical_flow
        from pycvcam import get_lena_image

        # Create two example images
        image1 = get_lena_image() # numpy array of shape (474, 474)

        # Shift the image to create a second image
        image2 = numpy.roll(image1, shift=5, axis=1) # Shift right by 5 pixels
        image2 = numpy.roll(image2, shift=3, axis=0) # Shift down by 3 pixels

        # Compute optical flow
        flow_x, flow_y = compute_optical_flow(image1, image2)

        # Check if all flow values in the center of the image are approximately (5, 3)
        valid_x = numpy.isclose(flow_x[100:300, 100:300], 5, atol=0.5)
        valid_y = numpy.isclose(flow_y[100:300, 100:300], 3, atol=0.5)
        valid = valid_x & valid_y
        assert valid.all(), "Flow values in the center of the image should be approximately (5, 3)"

    """
    # Input validation
    image1 = numpy.asarray(image1)
    image2 = numpy.asarray(image2)

    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same shape.")
    if not numpy.issubdtype(image1.dtype, numpy.unsignedinteger):
        raise TypeError("Input images must be of unsigned integer type.")
    if not numpy.issubdtype(image2.dtype, numpy.unsignedinteger):
        raise TypeError("Input images must be of unsigned integer type.")
    if image1.ndim not in [2, 3]:
        raise ValueError("Input images must be 2D or 3D arrays.")

    use_estimate_flow = False
    if estimate_flow_x is not None:
        use_estimate_flow = True
        estimate_flow_x = numpy.asarray(estimate_flow_x)
        if estimate_flow_x.shape != image1.shape[:2]:
            raise ValueError(
                "estimate_flow_x must have shape compatible with the input images."
            )

    if estimate_flow_y is not None:
        use_estimate_flow = True
        estimate_flow_y = numpy.asarray(estimate_flow_y)
        if estimate_flow_y.shape != image1.shape[:2]:
            raise ValueError(
                "estimate_flow_y must have shape compatible with the input images."
            )

    if use_estimate_flow and (estimate_flow_x is None or estimate_flow_y is None):
        raise ValueError(
            "Both estimate_flow_x and estimate_flow_y must be provided if using initial "
            "flow estimates."
        )

    if region is not None:
        x, y, w, h = region
        if not (
            isinstance(x, int)
            and isinstance(y, int)
            and isinstance(w, int)
            and isinstance(h, int)
        ):
            raise TypeError("Region coordinates and size must be integers.")
        if x < 0 or y < 0 or x + w > image1.shape[1] or y + h > image1.shape[0]:
            raise ValueError("Region is out of image bounds.")
        w_slice = slice(x, x + w)
        h_slice = slice(y, y + h)
    else:
        w_slice = slice(0, image1.shape[1])
        h_slice = slice(0, image1.shape[0])

    if not isinstance(channel, int):
        raise TypeError("Channel must be an integer.")
    if image1.ndim == 3 and (channel < 0 or channel >= image1.shape[2]):
        raise ValueError("Channel index out of bounds.")

    if disflow_params is not None and not isinstance(disflow_params, dict):
        raise TypeError("disflow_params must be a dictionary.")
    if disflow_params is None:
        disflow_params = {}

    # Create DIS optical flow object
    dis_optical_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    dis_optical_flow.setFinestScale(disflow_params.get("FinestScale", 0))
    dis_optical_flow.setVariationalRefinementAlpha(
        disflow_params.get("VariationalRefinementAlpha", 1.0)
    )
    dis_optical_flow.setVariationalRefinementDelta(
        disflow_params.get("VariationalRefinementDelta", 1.0)
    )
    dis_optical_flow.setVariationalRefinementGamma(
        disflow_params.get("VariationalRefinementGamma", 1.0)
    )
    dis_optical_flow.setVariationalRefinementEpsilon(
        disflow_params.get("VariationalRefinementEpsilon", 0.02)
    )
    dis_optical_flow.setVariationalRefinementIterations(
        disflow_params.get("VariationalRefinementIterations", 10)
    )
    dis_optical_flow.setUseMeanNormalization(
        disflow_params.get("UseMeanNormalization", True)
    )
    dis_optical_flow.setGradientDescentIterations(
        disflow_params.get("GradientDescentIterations", 500)
    )
    dis_optical_flow.setUseSpatialPropagation(
        disflow_params.get("UseSpatialPropagation", True)
    )
    dis_optical_flow.setPatchSize(disflow_params.get("PatchSize", 50))
    dis_optical_flow.setPatchStride(disflow_params.get("PatchStride", 10))

    # Extract the specified channel
    if image1.ndim == 3:
        img1_channel = image1[h_slice, w_slice, channel]
        img2_channel = image2[h_slice, w_slice, channel]
    else:
        img1_channel = image1[h_slice, w_slice]
        img2_channel = image2[h_slice, w_slice]

    # Convert images to uint8
    image1_range = numpy.iinfo(img1_channel.dtype).max
    image2_range = numpy.iinfo(img2_channel.dtype).max
    uint8_range = numpy.iinfo(numpy.uint8).max

    # Scale images to uint8
    alpha1 = uint8_range / image1_range if image1_range > uint8_range else 1.0
    alpha2 = uint8_range / image2_range if image2_range > uint8_range else 1.0

    img1_float64 = img1_channel.astype(numpy.float64) * alpha1
    img2_float64 = img2_channel.astype(numpy.float64) * alpha2
    img1_uint8 = numpy.round(numpy.clip(img1_float64, 0, uint8_range)).astype(
        numpy.uint8
    )
    img2_uint8 = numpy.round(numpy.clip(img2_float64, 0, uint8_range)).astype(
        numpy.uint8
    )

    # Compute optical flow
    estimate_flow = (estimate_flow_x, estimate_flow_y) if use_estimate_flow else None
    flow = dis_optical_flow.calc(img1_uint8, img2_uint8, estimate_flow).astype(
        numpy.float64
    )

    flow_x = numpy.full(image1.shape[:2], numpy.nan, dtype=numpy.float64)
    flow_y = numpy.full(image1.shape[:2], numpy.nan, dtype=numpy.float64)
    flow_x[h_slice, w_slice] = flow[:, :, 0]
    flow_y[h_slice, w_slice] = flow[:, :, 1]
    return flow_x, flow_y


def display_optical_flow(
    image: ArrayLike,
    flow_x: ArrayLike,
    flow_y: ArrayLike,
    region: Optional[Tuple[Integral, Integral, Integral, Integral]] = None,
    display_region: Optional[Tuple[Integral, Integral, Integral, Integral]] = None,
    alpha: Real = 0.5,
    channel: Integral = 0,
    norm_cmap: str = "inferno",
    comp_cmap: str = "bwr",
    norm_vmin: Optional[Real] = None,
    norm_vmax: Optional[Real] = None,
    comp_vmin: Optional[Real] = None,
    comp_vmax: Optional[Real] = None,
) -> None:
    """
    Display the optical flow overlaid on the given image using Matplotlib.

    Parameters
    ----------
    image: ArrayLike
        The background image with shape :math:`(H, W)` or :math:`(H, W, C)`.

    flow_x: ArrayLike
        The x-component of the optical flow (horizontal displacement) in pixels with
        shape :math:`(H, W)`.

    flow_y: ArrayLike
        The y-component of the optical flow (vertical displacement) in pixels with
        shape :math:`(H, W)`.

    region: Optional[Tuple[Integral, Integral, Integral, Integral]], optional
        A tuple specifying the region of interest in the format (x, y, width, height).
        If None, the entire image is displayed. Default is None.

    display_region: Optional[Tuple[Integral, Integral, Integral, Integral]], optional
        A tuple specifying the region of the flow to display in the format
        (x, y, width, height).
        If None, the entire flow is displayed. Default is None.

    alpha: Real, optional
        The alpha blending value for overlaying the optical flow on the image.
        Default is 0.5.

    channel: Integral, optional
        The channel of the image to display if it is multi-channel. Default is 0.

    norm_cmap: :class:`str`, optional
        The colormap for the flow magnitude. Default is 'inferno'.

    comp_cmap: :class:`str`, optional
        The colormap for the flow components. Default is 'bwr'.

    norm_vmin: Optional[Real], optional
        Minimum value for flow magnitude colormap normalization. Default is None.

    norm_vmax: Optional[Real], optional
        Maximum value for flow magnitude colormap normalization. Default is None.

    comp_vmin: Optional[Real], optional
        Minimum value for flow component colormap normalization. Default is None.

    comp_vmax: Optional[Real], optional
        Maximum value for flow component colormap normalization. Default is None.


    Returns
    -------
    None
        Displays the optical flow overlay on the image.


    See Also
    --------
    pycvcam.compute_optical_flow
        Compute the optical flow between two images using the DIS method of OpenCV.


    Examples
    --------
    Create two example images and compute the optical flow between them.

    .. figure:: /_static/textures/lena_texture.png
        :align: center
        :width: 50%

        Lena image used for the example.

    .. code-block:: python
        :linenos:

        import numpy
        import cv2
        from pycvcam import compute_optical_flow, display_optical_flow
        from pycvcam import get_lena_image

        # Create two example images
        image1 = get_lena_image() # numpy array of shape (474, 474)

        # cv2 distortion to create a second image
        image2 = cv2.undistort(image1, cameraMatrix=numpy.array(
            [[300, 0, 237], [0, 300, 237], [0, 0, 1]]
        ), distCoeffs=numpy.array([-0.2, 0.1, 0, 0]))

        # Compute optical flow
        flow_x, flow_y = compute_optical_flow(image1, image2)

        # Select a region of interest (optional)
        display_region = (10, 10, image1.shape[1]-20, image1.shape[0]-20) # (x, y, width, height)

        # Display the optical flow
        display_optical_flow(image1, flow_x, flow_y, display_region=display_region)

    """
    # Input validation
    image = numpy.asarray(image)
    flow_x = numpy.asarray(flow_x)
    flow_y = numpy.asarray(flow_y)

    if image.shape[:2] != flow_x.shape or image.shape[:2] != flow_y.shape:
        raise ValueError("Image and flow components must have compatible shapes.")
    if not numpy.issubdtype(image.dtype, numpy.unsignedinteger):
        raise TypeError("Input image must be of unsigned integer type.")
    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be 2D or 3D array.")

    if channel is not None:
        if not isinstance(channel, int):
            raise TypeError("Channel must be an integer.")
        if image.ndim == 3 and (channel < 0 or channel >= image.shape[2]):
            raise ValueError("Channel index out of bounds.")

    if region is not None:
        x, y, w, h = region
        if not (
            isinstance(x, int)
            and isinstance(y, int)
            and isinstance(w, int)
            and isinstance(h, int)
        ):
            raise TypeError("Region coordinates and size must be integers.")
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            raise ValueError("Region is out of image bounds.")
        w_slice = slice(x, x + w)
        h_slice = slice(y, y + h)
    else:
        w_slice = slice(0, image.shape[1])
        h_slice = slice(0, image.shape[0])

    if display_region is not None:
        fx, fy, fw, fh = display_region
        if not (
            isinstance(fx, int)
            and isinstance(fy, int)
            and isinstance(fw, int)
            and isinstance(fh, int)
        ):
            raise TypeError("Flow region coordinates and size must be integers.")
        if fx < 0 or fy < 0 or fx + fw > image.shape[1] or fy + fh > image.shape[0]:
            raise ValueError("Flow region is out of image bounds.")
        fw_slice = slice(max(fx, w_slice.start), min(fx + fw, w_slice.stop))
        fh_slice = slice(max(fy, h_slice.start), min(fy + fh, h_slice.stop))
    else:
        fw_slice = w_slice
        fh_slice = h_slice

    # Extract the useful channel
    if image.ndim == 3:
        img_channel = image[h_slice, w_slice, channel]
    else:
        img_channel = image[h_slice, w_slice]

    # Set the flow to nan outside the specified flow region
    flow_x_display = numpy.full_like(flow_x, numpy.nan)
    flow_y_display = numpy.full_like(flow_y, numpy.nan)
    flow_x_display[fh_slice, fw_slice] = flow_x[fh_slice, fw_slice]
    flow_y_display[fh_slice, fw_slice] = flow_y[fh_slice, fw_slice]

    # Extract the flow in the specified region
    flow_x_region = flow_x_display[h_slice, w_slice]
    flow_y_region = flow_y_display[h_slice, w_slice]

    # Compute flow magnitude
    flow_magnitude = numpy.sqrt(flow_x_region**2 + flow_y_region**2)

    # Plotting
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img_channel, cmap="gray")
    fplot = ax.imshow(
        flow_magnitude, cmap=norm_cmap, alpha=alpha, vmin=norm_vmin, vmax=norm_vmax
    )
    ax.set_title(r"Flow Magnitude $\|\mathbf{F}\|$ (pixels)")
    fig.colorbar(fplot, ax=ax)

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(img_channel, cmap="gray")
    fxplot = ax.imshow(
        flow_x_region, cmap=comp_cmap, alpha=alpha, vmin=comp_vmin, vmax=comp_vmax
    )
    ax.set_title(r"Flow X Component $F_x$ (pixels)")
    fig.colorbar(fxplot, ax=ax)

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(img_channel, cmap="gray")
    fyplot = ax.imshow(
        flow_y_region, cmap=comp_cmap, alpha=alpha, vmin=comp_vmin, vmax=comp_vmax
    )
    ax.set_title(r"Flow Y Component $F_y$ (pixels)")
    fig.colorbar(fyplot, ax=ax)

    plt.tight_layout()
    plt.show()
