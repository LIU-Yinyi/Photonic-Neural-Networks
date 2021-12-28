from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    readme_filedata = fh.read()


setup(
  name = 'pnn',         # How you named your package folder (pnn)
  packages = find_packages(),   # Chose the same as "name"
  version = '0.0.5',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Libraries for Photonic Neural Networks',   # Give a short description about your library
  long_description = readme_filedata,
  long_description_content_type = 'text/markdown',
  author = 'Yinyi',                   # Type in your name
  author_email = 'support@liuyinyi.com',      # Type in your E-Mail
  url = 'https://github.com/LIU-Yinyi/Photonic-Neural-Networks',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/LIU-Yinyi/Photonic-Neural-Networks/archive/refs/tags/v0.1.0-alpha.tar.gz',    # I explain this later on
  keywords = ['optic', 'photonic', 'neural networks', 'decomposition'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'scipy',
          'sympy',
          'matplotlib'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)