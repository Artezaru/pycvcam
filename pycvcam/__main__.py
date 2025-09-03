import argparse
import sys

from .visualizers import ZernikeDistortionVisualizerUI
from PyQt5.QtWidgets import QApplication

def __main__() -> None:
    r"""
    Main entry point of the package.

    This method contains the script to run if the user enter the name of the package on the command line. 

    .. code-block:: console

        pycvcam
        
    """
    raise NotImplementedError("The main entry point is not implemented yet - use pycvcam-gui for visualizations.")

def __main_gui__() -> None:
    r"""
    Graphical user interface entry point of the package.

    This method contains the script to run if the user enter the name of the package on the command line with the ``gui`` extension.

    .. code-block:: console

        pycvcam-gui
        
    """
    parser = argparse.ArgumentParser(
        description="pycvcam command-line interface"
    )
    parser.add_argument(
        "-zernike",
        action="store_true",
        help="Launch the zernike Distortion GUI"
    )
    args = parser.parse_args()

    if args.zernike:
        app = QApplication(sys.argv)
        window = ZernikeDistortionVisualizerUI()
        window.resize(1800, 800)
        window.show()
        sys.exit(app.exec_())
    else:
        parser.print_help()
        sys.exit(0)

