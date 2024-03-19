from setuptools import setup, find_packages

setup(
    name="policy_eval",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "gym==0.23.0",
        "mujoco-py==2.1.2.14",
        "tqdm==4.66.1",
        "pandas==2.0.3",
        "numpy==1.21.5",
        "tensorflow==2.6.0",
        "tensorflow-addons==0.16.1",
        "tensorflow-probability==0.14.1",
        "h5py==3.1.0",
        "typing-extensions==3.7.4.3",
        "tf-agents==0.9.0"
    ]
)
