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
 - Performance about  MAP  TOP1  TOP3  TOP5  TOP10    VEHICLE-DATASET: VeRi-776
   
   |  Network  |  MAP   |  TOP1  |  TOP3  |  TOP5  |  TOP10  | 
   | :-------: | :----: | :----: | :----: | :----: | :-----: |
   | SparkAct  | 0.8103 | 0.9613 | 0.9774 | 0.9815 | 0.9905  |       
   | Fighter   | 0.8240 | 0.9666 | 0.9785 | 0.9851 | 0.9923  | 
   | PowerNet  | 0.8194 | 0.9636 | 0.9774 | 0.9833 | 0.9911  | 
   | ResNet50  | 0.6235 | 0.8713 | 0.9172 | 0.9398 | 0.9684  | 
    
   
 - Performance about  MAP  TOP1  TOP3  TOP5  TOP10    VEHICLE-DATASET: VehicleID
   
   |  Network  |  MAP   |  TOP1  |  TOP3  |  TOP5  |  TOP10  | 
   | :-------: | :----: | :----: | :----: | :----: | :-----: |
 
   
 - Performance about  MAP  TOP1  TOP3  TOP5  TOP10    PERSON-DATASET: MSMT17
   
   |  Network  |  MAP   |  TOP1  |  TOP3  |  TOP5  |  TOP10  | 
   | :-------: | :----: | :----: | :----: | :----: | :-----: |

   
 - Performance about  MAP  TOP1  TOP3  TOP5  TOP10    PERSON-DATASET: Market1501
   
   |  Network  |  MAP   |  TOP1  |  TOP3  |  TOP5  |  TOP10  | 
   | :-------: | :----: | :----: | :----: | :----: | :-----: |
   | Flydream  | 0.8654 | 0.9495 | 0.9718 | 0.9786 | 0.9899  |       
   | Flydragon | 0.8732 | 0.9516 | 0.9736 | 0.9795 | 0.9869  |  
   | Sparck_CN | 0.8704 | 0.9477 | 0.9700 | 0.9780 | 0.9860  |
   | SparkPower | 0.8634 | 0.9391 | 0.9662 | 0.9742 | 0.9855  | 


   
