import setuptools
# Requires TensorFlow Datasets
setuptools.setup(
    install_requires=[
        'tensorflow_datasets==1.3.0',
    ],
    packages=setuptools.find_packages())
