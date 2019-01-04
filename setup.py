from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


setup(name='ShortStack',
      version='0.1.1',
      description='NanoString Hyb&Seq sequencing  and variant calling',
      url=' https://github.com/summerela/ShortStack',
      author='Summer R. Elasady',
      author_email='selasady@nanostring.com',
      license='MIT',
      packages=['ShortStack'],
      scripts=["bin/blat.py", "bin/S6_converter.py"],
      zip_safe=True,
      classifiers=['Programming Language :: Python :: 3.6'],
      install_requires=[
           'biopython==1.72',
           'configparser==3.5.0',
           'Cython==0.28.2',
           'dask==0.19.1',
           'ipywidgets==7.4.2',
           'networkx==2.1',
           'numba==0.38.1',
           'numpy==1.14.2',
           'pandas==0.22.0',
           'pandasql==0.7.3',
           'psutil==5.4.6',
           'pytest==4.0.2'
           'scikit-allel==1.1.10',
           'seqlog==0.3.9',
           'swifter==0.223',
           'ujson==1.35'
      ])
