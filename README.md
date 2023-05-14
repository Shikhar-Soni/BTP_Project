# BTP_Project

The repository contains the details of both the semesters of BTP Project (CP302 & CP303).

# To create a tmux (you can run code indefinitely here)

To check for all existing tmux sessions
* tmux ls

To create a new tmux session (index 1)
* tmux attach-session -t 1

To kill a tmux session with 1 index
* tmux kill-session -t 1


# The folder ./Diffusion_Model contains all the files associated with the SR3 model used, along with all the code modifications made.

Install the requirements using:
* pip install -r requirement.txt

* cd Diffusion_Model
To train the diffusion model, run the below command (hyper parameters are supplied by the config files present in ./Diffusion_Model/config)
*  python3 sr.py -p train -c config/sr_sr3_64_512.json

# The folder ./W-Net contains all the files associated with the W-Net, W-Net Combined and W-Net 3-Layer (our change to the original W-Net architecture)

install keras
* pip install keras

You can find the jupyter notebooks associated with each model type in ./W-Net/JNotebooks

# Find all the results related to variations of SR3 in ./Diffusion-Results

The report containing all the data and detailed information is attached as "MRI_enhancements_using_neural_networks.pdf"

# All the code related to augmenting the datasets can be found in ./datasets
