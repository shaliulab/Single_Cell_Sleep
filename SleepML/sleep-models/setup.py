from setuptools import setup, find_packages

# attention. you need to update the numbers ALSO in the imgstore/__init__.py file
version = "1.1.1"

PACKAGE_NAME = "sleep_models"
with open(f"{PACKAGE_NAME}/_version.py", "w") as fh:
    fh.write(f"__version__ = '{version}'\n")

# This call to setup() does all the work
setup(
    name=PACKAGE_NAME,
    version=version,
    # description="High resolution monitoring of Drosophila",
    # long_description=README,
    # long_description_content_type="text/markdown",
    ##url="https://github.com/realpython/reader",
    # author="Antonio Ortega",
    # author_email="antonio.ortega@kuleuven.be",
    # license="MIT",
    # classifiers=[
    #    "License :: OSI Approved :: MIT License",
    #    "Programming Language :: Python :: 3",
    #    "Programming Language :: Python :: 3.7",
    # ],
    packages=find_packages(),
    # include_package_data=True,
    install_requires=[
        "anndata==0.7.5",
        "scanpy==1.6.0",
        "numpy==1.21.5",
        "umap-learn==0.5.0",
        "tensorflow==2.0.0",
        "torch",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "joblib",
        "opencv-python",
        "Pillow",
        "interpret",
        "wandb",
        "openpyxl",
        "pyaml",
        "scikit-learn",
        "seaborn",
    ],
    entry_points={
        "console_scripts": [
            "make-dataset=sleep_models.bin.make_dataset:main",
            "train-model=sleep_models.bin.train_model:main",
            "train-models=sleep_models.bin.train_models:main",
            "crosspredict=sleep_models.bin.crosspredict:main",
            "make-matrixplot=sleep_models.bin.make_matrixplot:main",
            "make-umapplot=sleep_models.bin.make_umapplot:main",
            "get-marker-genes=sleep_models.bin.get_marker_genes:main",
            "remove-marker-genes=sleep_models.bin.remove_marker_genes:main",
            "sleep-models-sweep=sleep_models.bin.sweep:main",
            "train-torch-model=sleep_models.bin.train_torch_model:main",
            "test-torch-model=sleep_models.bin.test_torch_model:main",
        ]
    },
    python_requires='>=3.7.4',
)
