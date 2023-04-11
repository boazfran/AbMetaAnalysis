from setuptools import setup, find_packages

setup(
    name='AbMetaAnalysis',
    version='1.0.4',
    packages=find_packages(),
    url='https://github.com/boazfran/AbMetaAnalysis',
    license='MIT (X11)',
    author='Boaz Frankel',
    author_email='boazfr@gmail.com',
    description='Package for Meta analysis of BCR repertoires ',
    install_requires=[
        'changeo>=1.0.2',
        'ipython>=8.10.0',
        'matplotlib>=3.6.2',
        'numpy>=1.23.5',
        'pandas>=1.5.3',
        'psutil>=5.9.4',
        'ray>=2.2.0',
        'scikit_learn>=1.2.1',
        'scipy>=1.10.1',
        'seaborn>=0.12.2',
        'statsmodels>=0.13.5',
        'protobuf==3.19.*'
    ]
)
