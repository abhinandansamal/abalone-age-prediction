import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "abalone-age-prediction"
AUTHOR_USER_NAME = "abhinandansamal"
SRC_REPO = "abalone-age-prediction "
AUTHOR_EMAIL = "samalabhinandan06@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package to predict the age of abalone from various physical measurements.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "numpy==1.25.2",
        "pandas==2.0.3",
        "matplotlib==3.7.1",
        "seaborn==0.13.1",
        "scipy==1.11.4",
        "scikit-learn==1.2.2",
        "joblib==1.4.2",
        "PyYAML==6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)