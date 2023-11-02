from setuptools import setup
from distutils.cmd import Command
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

import glob
import os


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    PY_CLEAN_DIRS = [
        "./build",
        "./dist",
        "./*.egg-info"
    ]
    description = "Custom clean command to tidy up the project root"
    user_options = []
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        for path_spec in self.PY_CLEAN_DIRS:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(dir_path, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(dir_path):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("{0} is not a path inside {1}".format(path, dir_path))
                print("Removing {}".format(os.path.relpath(path)))
                os.system("rm -rf {}".format(path))


setup(
    name='point_ops',
    version='v1.0',
    description='deep layers used to convert between point and voxels',
    install_requires=['PyYAML>=5.4.1', 'scipy>=1.3.1', 'nuscenes-devkit'],
    ext_modules=[
       CUDAExtension(name = 'point_ops.pytorch_plugins',
            sources = ['src/pt_plugin.cpp', 'src/pt_plugin_cuda_kernel.cu'],
            include_dirs = ['src'])],
    cmdclass={
        "build_ext": BuildExtension,
        "clean": CleanCommand
    }
)
