from setuptools import setup, find_packages

setup(
    name="vit-pytorch-robust",
    packages=find_packages(exclude=["examples"]),
    version="0.0.1",
    license="MIT",
    description="Vision Transformer (ViT) - Pytorch - noise robust",
    long_description_content_type="text/markdown",
    author="Randall Balestriero",
    author_email="randallbalestriero@gmail.com",
    url="https://github.com/lucidrains/vit-pytorch",
    keywords=[
        "artificial intelligence",
        "attention mechanism",
        "image recognition",
        "noise robust",
    ],
    install_requires=["einops>=0.6.0", "torch>=1.10", "torchvision"],
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=["pytest", "torch==1.12.1", "torchvision==0.13.1"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
