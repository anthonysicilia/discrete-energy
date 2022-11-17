import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='discrete_energy',
    version='0.0.1',
    author='Anthony Sicilia',
    author_email='anthonysicilia.contact@gmail.com',
    description='Compute discrete energy distance.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/anthonysicilia/discrete-energy',
    project_urls = {
        "Bug Tracker": "https://github.com/anthonysicilia/discrete-energy/issues"
    },
    license='MIT',
    packages=['discrete_energy'],
    install_requires=['scikit_learn', 'torch'],
)