# Human_Pose_Estimation_Flask_App
Flask application for human pose estimation using webcam of the computer.

This repository contains the flask application to test the models from [Soft_Gated_Pose_Estimation_Pytorch](https://github.com/dkurzend/Soft_Gated_Pose_Estimation_Pytorch) in real-time.

### Supportd Models
- [x] Stacked Hourglass Network
- [x] Soft-Gated Skip Connections


## Getting Started

### Requirements
- Python 3.10.4
- Pytorch


### Installation
1. Clone the repository:

    ```
    git clone https://github.com/dkurzend/Human_Pose_Estimation_Flask_App.git
    ```
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and create a virtual environment.
    ```
    conda create --name hpe_app
    ```

3. Activate the virtual env and install pip as well as the dependencies.
    ```
    conda activate hpe_app
    conda install pip
    pip install -r requirements.txt
    ```
    (Alternatively use venv instead of miniconda)

4. Download the models and put them into the `models` folder ([soft-gated skip connections](https://matix.li/4704c467c50a), [stacked hourglass](https://matix.li/9680ec1eb999)). You have to download both models.


The `requirements.txt` file includes the cpu version of pytorch. If your computer/laptop has a gpu available feel free to change the pytorch version to one including cuda (tested with cuda version 11.6). If cuda is available, you will be able to switch between gpu and cpu in the application.


## Result

|                      	| **Soft-Gated Skip Connections** 	| **Stacked Hourglass Network** 	|
|:--------------------:	|:-------------------------------:	|:-----------------------------:	|
| Number of Parameters 	|             13.6 Mio            	|            32.8 Mio           	|
|     Speed on CPU*    	|       0.54 sec (1.89 fps)*      	|      0.97 sec (1.04 fps)*     	|
|     Speed on GPU*    	|       0.09 sec (11.1 fps)*      	|      0.18 sec (5.56 fps)*     	|
|    Speed on CPU**    	|      0.24 sec (4.17 fps)**      	|     0.42 sec (2.38 fps)**     	|
|    Speed on GPU**    	|    **0.03 sec (33.3 fps)****    	|   **0.06 sec (16.67 fps)****  	|

<h6>*System: Laptop with i7-10750H CPU and GeForce RTX 2060 GPU </h6>
<h6>**System: Desktop with  i9-9900K CPU and GeForce RTX 2080 GPU </h6>

<br>


![Example prediction](/assets/demo_screenshot.png)         |  ![Example prediction](/assets/Bild1.png)
:-------------------------:|:-------------------------:







## Final Note
This repository was part of a university project at university of Tübingen.
Project team:<br>
David Kurzendörfer, Jan-Patrick Kirchner, Tim Herold, Daniel Banciu
