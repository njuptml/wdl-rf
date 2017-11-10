from __future__ import absolute_import
from distutils.core import setup

setup(name='wdl',
      version='1.0.0',
      description='Computes differentiable activity values of moleculars.',
      author='Jiansheng Wu and Qiuming Zhang',
      author_email="jansen@njupt.edu.cn, jansen@njupt.edu.cn",
      packages=['wdl'],
      install_requires=['numpy>=1.8', 'scipy>=0.15', 'autograd'],
      keywords=['weight neural Fingerprints','neural networks','Molecular Fingerprints',
                'Morgan fingerprints','machine learning','Python', 'Numpy', 'Scipy'],)
