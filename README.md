# YOLinO-magnetic_tape
## Generate ground truth tensors

### Clone the main repository and the submodules
   ```sh
   git clone https://github.com/manojlovska/YOLinO-magnetic_tape.git
   
   cd YOLinO-magnetic_tape/
   
   git submodule init
   git submodule update
   ```
### Install the requirements
   ```sh
   pip install -r Annotation-Conversion/requirements.txt
   ```
### Usage
   ```sh
   python3 toTensor.py
   ```
