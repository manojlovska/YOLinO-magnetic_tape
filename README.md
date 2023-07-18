# YOLinO-magnetic_tape
## Generate ground truth tensors

### Clone the main repository and the submodules
   ```sh
   git clone https://github.com/manojlovska/YOLinO-magnetic_tape.git
   ```
   ```sh
   cd YOLinO-magnetic_tape/
   ```
   ```sh
   git submodule init
   git submodule update
   ```
### Install the requirements
   ```sh
   pip install -r requirements.txt
   ```
### Usage
To generate the ground truth tensors run
   ```sh
   python3 toTensor.py
   ```
To visualize run
   ```sh
   python3 visualize.py
   ```
