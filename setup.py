from setuptools import setup

def readme():
      with open('README.rst') as f:
            return f.read()

setup(name='nn_generator',
      version='1.0.5',
      description='FC NN configurable via json/YML config file',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='configurable nn deep_nn fully connected',
      url='https://github.com/Alisa-lisa/nn_generator',
      author='Alisa Dammer',
      author_email='alisa.dammer@gmail.com',
      license='MIT',
      packages=['nn_generator'],
      include_package_data=True,
      install_requires=[
            'matplotlib',
            'numpy',
            'pyparsing',
            'pytz',
            'PyYAML',
      ],
      zip_safe=False)
