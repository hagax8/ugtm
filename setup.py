from setuptools import setup, find_packages
#from sphinx.setup_command import BuildDoc
#cmdclass = {'build_sphinx': BuildDoc}

setup(name='ugtm',
      version='1.1.4',
      description='Generative Topographic Mapping (GTM) for python, GTM classification and GTM regression',
      long_description=open('README.rst').read(),
      url='http://github.com/hagax8/ugtm',
      author='Helena A. Gaspar',
      author_email='hagax8@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['tests*']),
      install_requires=['numpy', 'sklearn',
                        'matplotlib', 'scipy', 'mpld3>=0.3'],
      test_suite='nose.collector',
      tests_require=['nose'],
      )
