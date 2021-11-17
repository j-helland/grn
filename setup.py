from setuptools import setup, find_packages

install_requires = [
    'nvidia-ml-py3',
    'grpcio',
    'depq',
    'rich',
]

dev_requires = [
    'grpcio-tools',
]

setup(
    name='gpu-resource-negotiator',
    packages=find_packages(),
    version='0.2.3',
    license='unlicense',
    description='GPU Resource Negotiator (GRN). Map N jobs to M devices on the same machine.',
    author='Jonathan Helland',
    url='https://github.com/j-helland/grn',
    download_url='https://github.com/j-helland/grn/archive/refs/tags/pre-v-0.2.3.tar.gz',
    keywords=['GPU', 'load balancing'],
    install_requires=install_requires,
    extras_require={ 'dev': dev_requires },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: The Unlicense (Unlicense)',
        'Programming Language :: Python :: 3',
    ],
    scripts=['bin/grn-start', 'bin/grn-run']
)