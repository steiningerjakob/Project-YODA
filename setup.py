from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.read().splitlines()
requirements = [x.strip() for x in content if 'git+' not in x and '#' not in x]
requirements = [r for r in requirements if len(r) > 0]

setup(name='projectYoda',
      version="1.0.1",
      description="Project Description",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/Project-YODA-run'],
      zip_safe=False)
