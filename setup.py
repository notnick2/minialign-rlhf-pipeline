from setuptools import setup, find_packages

setup(
    name="minialign",
    version="0.1.0",
    description="End-to-end RLHF alignment pipeline: annotation, reward modeling, PPO/DPO/GRPO training",
    author="MiniAlign Dev",
    author_email="dev@minialign.ai",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "trl>=0.7.0",
        "peft>=0.6.0",
        "gradio>=4.0.0",
        "anthropic>=0.20.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "datasets>=2.14.0",
        "pyyaml>=6.0",
        "sqlalchemy>=2.0.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "minialign-annotate=annotation.app:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
