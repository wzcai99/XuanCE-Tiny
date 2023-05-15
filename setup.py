from setuptools import setup,find_packages
import codecs
setup(
    name='xuance',
    packages=find_packages(),
    version='0.1.0',
    description= 'xuance: a simple and clean deep reinforcement learning framework and implementations',
    long_description=codecs.open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='MIT License',
    install_requires=[
        'gym==0.26.1',
        'matplotlib>=3.7.1',
        'opencv-python >= 4.7.0.72',
        'numpy>=1.23.1',
        'pandas>=1.5.3',
        'PyYAML>=6.0',
        'scipy>=1.10.1',
        'seaborn>=0.12.2',
        'tensorboard>=2.12.0',
        'torch==2.0.1',
        'torchvision==0.15.2',
        'tqdm>=4.65.0',
        'mujoco==2.3.3',            
        'mujoco-py==2.1.2.14',
        'free-mujoco-py==2.1.6',
        'dm_control==1.0.11',
    ]
)