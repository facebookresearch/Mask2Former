from setuptools import setup

setup(
    name="mask2former",
    package_dir={"mask2former": "mask2former"},
    version="1.0",
    install_requires=[
        "cython",
        "scipy",
        "shapely",
        # timm
        "h5py",
        "submitit",
        "scikit-image",
        "pandas",
        "pycocotools",
        "tqdm",
        "matplotlib",
        "torch",
        "numpy",
    ],
    extras_require={},
)
