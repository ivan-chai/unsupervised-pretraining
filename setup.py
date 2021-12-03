from setuptools import setup, find_namespace_packages

setup(
    version="0.1.0",
    name="ulib",
    long_description="Unsupervised pretraining library.",
    license="MIT",
    url="https://github.com/ivan-chai/unsupervised-pretraining",
    author="",
    author_email="",
    packages=find_namespace_packages(where="src"),
    python_requires='>=3.9',
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "ulib_evaluate = unsupervised_pretraining.scripts.evaluate:main"
        ]
    },
    install_requires=[
        "torch==1.10.0",
        "torchvision==0.11.1",
        "pytorch_lightning==1.5.1",
        "torchmetrics==0.6.0",
        "hydra-core==1.1.1",
        "omegaconf==2.1.1",
        "albumentations==1.1.0",
        "pytest==6.2.5"
    ]
)
