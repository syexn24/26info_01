from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "rummikub_env",
        ["rummikub_env.cpp"],
        define_macros=[
            ("RUMMIKUB_PYBIND", "1"),
        ],
        cxx_std=17,
    ),
]

setup(
    name="rummikub_env",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
