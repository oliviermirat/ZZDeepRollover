import os
import setuptools
from setuptools import setup


def read_file(file):
  with open(file) as f:
    return f.read()


setup(
  name = 'zzdeeprollover',
  version = read_file(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'zzdeeprollover', 'version.txt')).strip(),
  license='AGPL-3.0',
  description = 'Detect rollovers in zebrafish larvae',
  long_description=read_file(os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md")),
  long_description_content_type='text/markdown',
  author = 'Olivier Mirat',
  author_email = 'olivier.mirat.om@gmail.com',
  url = 'https://github.com/oliviermirat/ZZDeepRollover',
  download_url = 'https://github.com/oliviermirat/ZZDeepRollover/releases/latest',
  keywords = ['Animal', 'Behavior', 'Tracking', 'Zebrafish', 'Deep Learning', 'Rolling'],
  install_requires=[
    "scikit-learn",
    "torch",
    "numpy",
    "matplotlib",
    "torchvision ",
    "pandas",
    "pillow",
    "opencv-python-headless",
  ],
  packages=setuptools.find_packages(),
  data_files=[
    (
      "zzdeeprollover",
      [
        "zzdeeprollover/version.txt"
      ],
    )
  ],
  include_package_data=True,
  classifiers=[
    'Programming Language :: Python :: 3'
  ],
)
