from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="karaokeai",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Neural Voice-to-MIDI Transcription System using Whisper and Transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/KaraokeAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "karaokeai-train=train:main",
            "karaokeai-inference=inference:main",
            "karaokeai-midify=frontend.midify_app:main",
            "karaokeai-karaoke=frontend.karaoke_app:main",
        ],
    },
) 