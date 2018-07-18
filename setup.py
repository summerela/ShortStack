from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


setup(name='ShortStack',
      version='0.1.1',
      description='NanoString assembly and variant calling algorithm',
      url=' https://github.com/summerela/ShortStack',
      author='Summer R. Elasady',
      author_email='selasady@nanostring.com',
      license='MIT',
      packages=['ShortStack', 'ShortStack.test'],
      scripts=["bin/blat.py", "bin/S6_converter.py"],
      zip_safe=True,
      classifiers=['Programming Language :: Python :: 3.6'],
      install_requires=[
          'pandas >= 0.22.0',
          'numpy >= 1.14.2',
          'pandasql', 
          'Cython',
          'seqlog', 
          'psutil', 
          'scipy >= 1.0.1',
          'scikit-allel',
          'matplotlib >= 2.2.2'
      ])