from setuptools import setup

install_requires = [
    'nvidia-ml-py3',
    'grpcio',
    'grpcio-tools',
    'rich',
]

setup(
    name='glb',
    packages=['glb'],
    version='0.1',
    license='unlicense',
    description='GPU Load Balancer (GLB). Map N jobs to M devices on the same machine.',
    author='Jonathan Helland',
    url='https://github.com/j-helland/glb',
    download_url='https://github.com/j-helland/glb/archive/v_01.tar.gz',
    keywords=['GPU', 'load balance'],
    install_requires=install_requires,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Machine Learning :: Training Tools',
        'License :: OSI Approved :: The Unlicense',
        'Programming Language :: Python :: 3',
    ],
    entry_points={
        'console_scripts': [
            'glb-start=glb.scripts.glb_start:start',
            'glb-run=glb.scripts.glb_run:run']
    },
)