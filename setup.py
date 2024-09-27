from setuptools import setup

setup(
    name='My_Library',
    version='1.0.0',
    description='This library imitates torch library and also implements autodifferentiation.',
    author='Naol Samuel Erega',
    author_email='samuelnaol7@gmail.com',
    url='https://github.com/samuelnaol7/My_Library',
    packages=['My_Library'],
    install_requires=[
        'numpy',
        # other dependencies
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
