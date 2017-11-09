# ---------------------------------------------------------------
# ------------- Install Windows Software ------------------------
# ---------------------------------------------------------------
# On Windows:
# Install WSL (Ubuntu)
# You must be on the lastest version and update of windows, so before you do anything check for updates.
# This may be helpful https://msdn.microsoft.com/en-us/commandline/wsl/install_guide
# NOTE: Never edit a Linux file with Windows. It is OK to edit Windows files with Linux.

# Install XServer
# You need an Xserver to display GUI components
# Xming seems to work well https://sourceforge.net/projects/xming/
# ---------------------------------------------------------------
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# ------------- Install Linux Software --------------------------
# ---------------------------------------------------------------
# On Linux:
# Install Anaconda, Tensorflow, OpenAI Gym, Jupyter Notebook
# These websites may be useful 
# https://medium.com/xtrememl/why-how-to-use-windows-10-wsl-built-in-linux-for-machine-learning-6a225f4bbd3a
# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/16_Reinforcement_Learning.ipynb
# https://www.tensorflow.org/install/install_linux
# https://github.com/openai/gym
# Do the following in a bash terminal (run your new Ubuntu program)
# Download Anaconda (Latest version doesn't work with WSL, so use this one)
wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
# Install Anaconda
sudo bash Anaconda3-4.4.0-Linux-x86_64.sh
conda info
# Update Anaconda
sudo chmod -R 777 anaconda3
conda update --all --y
# Create Tensorflow Environment
conda create -n tensorflow
source activate tensorflow
# Install TensorFlow
conda install tensorflow
# Create OpenAI Gym environment
conda create --name tf-gym --clone tensorflow
source activate tf-gym
# Install all dependencies
conda install matplotlib scipy
#sudo apt-get install build-essential # I don't think this is required
# Some of these should be included in Anaconda and thus shouldn't be installed through apt-get (still works if you install all of them)
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig 
pip install gym
pip install gym[atari]
# Install firefox
sudo bash -c "dbus-uuidgen > /var/lib/dbus/machine-id"
sudo apt-get install -y firefox
# Dependencies for Jupyter Notebook
conda install -c anaconda ipykernel 
python -m ipykernel install --user --name tf-gym --display-name "Python (tf-gym)"
# ---------------------------------------------------------------
# ---------------------------------------------------------------



# ---------------------------------------------------------------
# ------------- Start Tutorial ----------------------------------
# ---------------------------------------------------------------
# Download tutorial
git clone https://github.com/Hvass-Labs/TensorFlow-Tutorials.git
# Download checkpoint data and get into home folder
# Faster to download from google
wget http://hvass-labs.org/projects/tensorflow/tutorial16/Breakout-v0.tar.gz
tar -xvzf Breakout-v0.tar.gz
# Now follow the tutorial
pip install prettytensor
jupyter notebook

python initialize.py
# ---------------------------------------------------------------
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# ------------- Quality of Life Improvements (optional) ---------
# ---------------------------------------------------------------
# Download Sublime
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
# Install Sublime and dependencies
sudo apt-get install -y apt-transport-https
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
sudo apt-get update
sudo apt-get install -y libgtk2.0-0 sublime-text

# .bashrc edits
echo '# Use X server to display graphics' >> ~/.bashrc
echo 'export DISPLAY=:0' >> ~/.bashrc
# Anaconda may have automatically done this already (don't run these if so)
echo '# Add Anaconda binaries to you path' >> ~/.bashrc
# Change your_username to...
echo 'export PATH=/home/your_username/anaconda3/bin:$PATH' >> ~/.bashrc

# Setup shared workspace between linux/windows (don't store linux files here)
ln -s /mnt/c/Users/dbowd/WSL_Workspace workspace
# ---------------------------------------------------------------
# ---------------------------------------------------------------