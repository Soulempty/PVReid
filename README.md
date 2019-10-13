# Vehicle and Person Reid

### Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/Soulempty/PVReid.git
   ```
2. Dataset Preparation: Market1501,MSMT17,VehicleID,VeRi-776
3. Train:

   - Specify the data path,model name, task type[person/vehicle],data name,gpu id

     ```shell
     python train.py --data_path ../data/Market-1501 --name flydragon --dtype person --data_name market --device 2
     ```
     ```shell
     python train.py --data_path ../vehicle/VeRi --name fighter --dtype vehicle --data_name veri --device 4

     ```
