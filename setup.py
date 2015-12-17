# (c) 2014-2015 Max Planck Society
# see accompanying LICENSE.txt file for licensing and contact information

from distutils.core import setup
from distutils.util import convert_path
import os


def _get_version():
    """"Convenient function to get the version of this package"""

    ns = {}
    version_path = convert_path('./version.py')
    if not os.path.exists(version_path):
        return None
    with open(version_path) as version_file:
        exec(version_file.read(), ns)

    return ns['__version__']


setup(name='livius',
      version=_get_version(),
      packages=['livius',
                'livius.audio',
                'livius.util',
                'livius.video',
                'livius.video.editing',
                'livius.video.processing',
                'livius.video.processing.jobs',
                'livius.video.processing.visualization',
                'livius.video.processing.vault',
                'livius.video.processing.vault.CMT'
                ],
      package_data={'livius': ['default_config.json',
                               'ressources/*.png',
                               'ressources/*.zip',
                               'ressources/credit_images/*.png']},
      ext_modules=[],
      author='Raffi Enficiaud, Parnia Bahar, Stephan Wenninger, Edgar Klenske',
      author_email='raffi.enficiaud@tuebingen.mpg.de',
      maintainer='Raffi Enficiaud',
      maintainer_email='raffi.enficiaud@tuebingen.mpg.de',
      url='http://is.tuebingen.mpg.de',
      install_requires=['numpy'],
      description='Livius: the conference video creator',
      license='Not licensed yet',
      )
