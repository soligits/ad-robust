from setuptools import setup, find_packages

setup(
    name="improved-diffusion",
    py_modules=["improved_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)

# setup(
#     name="ad-utils",
#     py_modules=["ad_utils"],
#     install_requires=find_packages(),
# )