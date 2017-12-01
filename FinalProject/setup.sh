# Update Ubuntu
cd ~
sudo apt-get update
sudo apt-get upgrade -y
# Install Anaconda
wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh -b
echo '# Add Anaconda binaries to you path' >> ~/.bashrc
echo 'export PATH=/home/$USER/anaconda3/bin:$PATH' >> ~/.bashrc
. .bashrc
conda update --all -y
# Install Tensorflow
conda create -n tensorflow python=3.6 -y
. activate tensorflow
sudo apt-get install -y libcupti-dev
#pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp36-cp36m-linux_x86_64.whl
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl
# Install OpenAI Gym
conda create --name tf-gym --clone tensorflow
. activate tf-gym
sudo apt-get install -y zlib1g-dev libboost-all-dev swig cmake libsdl2-dev
pip install 'gym[all]'
pip install keras
