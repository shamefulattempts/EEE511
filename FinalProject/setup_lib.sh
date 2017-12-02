#mv ~/anaconda3/envs/tf-gym/lib/python3.6/site-packages/gym/envs/classic_control/cartpole.py ~/anaconda3/envs/tf-gym/lib/python3.6/site-packages/gym/envs/classic_control/cartpole.py.bak
cp cartpole.py ~/anaconda3/envs/tf-gym/lib/python3.6/site-packages/gym/envs/classic_control/
#mv ~/anaconda3/envs/tf-gym/lib/python3.6/site-packages/gym/envs/__init__.py ~/anaconda3/envs/tf-gym/lib/python3.6/site-packages/gym/envs/__init__.py.bak
cp __init__.py ~/anaconda3/envs/tf-gym/lib/python3.6/site-packages/gym/envs/
