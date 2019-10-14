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
   | Flydragon  | 0.4189 | 0.4284 | 0.4284 | 0.4284 | 0.4284  |       
   | Flydream  | 0.4189 | 0.4305 | 0.4284 | 0.4284 | 0.4284  |      
   | Flydragon | 0.4189 | 0.4415 | 0.4284 | 0.4284 | 0.4284  |       
   | Flydragon | 0.4189 | 0.4414 | 0.4284 | 0.4284 | 0.4284  |     
   
 - Performance about  MAP  TOP1  TOP3  TOP5  TOP10    VEHICLE-DATASET: VehicleID
   
   |  Network  |  MAP   |  TOP1  |  TOP3  |  TOP5  |  TOP10  | 
   | :-------: | :----: | :----: | :----: | :----: | :-----: |
   | Flydragon | 0.4189 | 0.4284 | 0.4284 | 0.4284 | 0.4284  |       
   | Flydragon | 0.4189 | 0.4305 | 0.4284 | 0.4284 | 0.4284  |      
   | Flydragon | 0.4189 | 0.4415 | 0.4284 | 0.4284 | 0.4284  |       
   | Flydragon | 0.4189 | 0.4414 | 0.4284 | 0.4284 | 0.4284  | 
   
 - Performance about  MAP  TOP1  TOP3  TOP5  TOP10    PERSON-DATASET: MSMT17
   
   |  Network  |  MAP   |  TOP1  |  TOP3  |  TOP5  |  TOP10  | 
   | :-------: | :----: | :----: | :----: | :----: | :-----: |
   | Flydragon | 0.4189 | 0.4284 | 0.4284 | 0.4284 | 0.4284  |       
   | Flydragon | 0.4189 | 0.4305 | 0.4284 | 0.4284 | 0.4284  |      
   | Flydragon | 0.4189 | 0.4415 | 0.4284 | 0.4284 | 0.4284  |       
   | Flydragon | 0.4189 | 0.4414 | 0.4284 | 0.4284 | 0.4284  |
   
 - Performance about  MAP  TOP1  TOP3  TOP5  TOP10    PERSON-DATASET: Market1501
   
   |  Network  |  MAP   |  TOP1  |  TOP3  |  TOP5  |  TOP10  | 
   | :-------: | :----: | :----: | :----: | :----: | :-----: |
   | Flydream  | 0.8654 | 0.9495 | 0.9718 | 0.9786 | 0.9899  |       
   | Flydragon | 0.4189 | 0.4305 | 0.4284 | 0.4284 | 0.4284  |      
   | Flydragon | 0.4189 | 0.4415 | 0.4284 | 0.4284 | 0.4284  |       
   | Flydragon | 0.4189 | 0.4414 | 0.4284 | 0.4284 | 0.4284  |

   
