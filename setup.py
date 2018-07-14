from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='ShortStack',
      version='0.1',
      description='NanoString assembly and variant calling algorithm',
      url=' https://github.com/summerela/ShortStack',
      author='Summer R. Elasady',
      author_email='selasady@nanostring.com',
      license='MIT',
      packages=['ShortStack', 'ShortStack.test'],
      script=["bin/blat", "bin/S6_converter.py"],
      zip_safe=False,
      classifiers=['Programming Language :: Python :: 3.6'],
      install_requires=[
          'python  >= 3.6',
          'pandas >= 0.22.0',
          'numpy >= 1.14.2',
          'pandasql >= 0.7.3',
          'Cython >= 0.28.2',
          'seqlog >= 0.3.9',
          'psutil >= 5.4.6',
          'scipy >= 1.0.1',
          'scikit-allel >= 1.1.10',
          'matplotlib >= 2.2.2',
          'scipy >= 1.0.1'
      ])