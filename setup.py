from setuptools import setup, find_packages

setup(name='nn_generator',
      version='1.0.9',
      description='FC NN configurable via json/YML config file',
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
      packages=find_packages(exclude=['examples.*', 'examples']),
      include_package_data=True,
      install_requires=[
            'matplotlib',
            'numpy',
            'pyparsing',
            'pytz',
            'PyYAML',
      ],
      python_requires='>=3',
      zip_safe=False)
