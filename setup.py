from setuptools import setup, find_packages
#from sphinx.setup_command import BuildDoc
#cmdclass = {'build_sphinx': BuildDoc}

setup(name='ugtm',
      version='2.0.0',
      description='Generative Topographic Mapping (GTM) for python, GTM classification and GTM regression',
      long_description=open('README.rst').read(),
      url='http://github.com/hagax8/ugtm',
      author='Helena A. Gaspar',
      author_email='hagax8@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['tests*']),
      install_requires=['numpy>=1.14.5', 'scikit-learn>=0.20',
                        'matplotlib>=2.2.2', 'scipy>=0.19.1', 'mpld3>=0.3', 'jinja2>=2.10'],
      test_suite='nose.collector',
      tests_require=['nose'],
      )
