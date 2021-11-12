from setuptools import setup

install_requires = [
    'nvidia-ml-py3',
    'grpcio',
    'grpcio-tools',
    'rich',
]

setup(
    name='glb',
    install_requires=install_requires,
    packages=['glb'],
    entry_points={
        'console_scripts': [
            'glb-start=glb.scripts.glb_start:start',
            'glb-run=glb.scripts.glb_run:run']
    },
)