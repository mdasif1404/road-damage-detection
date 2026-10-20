# Road Damage Detection Using YOLOv8



A deep learning-based web application that detects road damage from images using YOLOv8 and Flask. The application displays annotated images and provides an estimated summary of detected road damage.



## Features



* Detect road damage using a trained YOLOv8 model.
* Identify three damage classes: potholes, cracks, and patches.
* Upload road images through a Flask web interface.
* View annotated images with detection results.
* Process multiple images using the batch-processing feature.
* Generate approximate road-damage mass and repair-cost estimates.



## Technologies Used



* **Programming:** Python, JavaScript, HTML, CSS
* **Deep Learning:** YOLOv8, Ultralytics, PyTorch
* **Computer Vision:** OpenCV
* **Web Framework:** Flask
* **Tools:** Git, GitHub, Google Colab



## Project Structure



```text

road-damage-detection/

├── models/

│   └── road\_damage\\\_best.pt

├── static/

│   └── css/

├── templates/

├── app.py

├── model.py

├── test\_model.py

├── train\_yolo.py

├── convert\_xml\_to\_yolo.py

├── data.yaml

├── requirements.txt

├── .gitignore

└── .gitattributes

```



## Installation and Setup



### 1\. Clone the repository



```bash

git clone https://github.com/mdasif1404/road-damage-detection.git

cd road-damage-detection

```



### 2\. Create and activate a virtual environment



```bash

python -m venv venv

```



Windows:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



### 3\. Install dependencies



```bash

pip install -r requirements.txt

```



### 4\. Download the trained model



The application requires the trained model at:



```text

models/road\_damage\\\_best.pt

```



The repository uses Git LFS for the model file. Ensure Git LFS is installed and the model file has been downloaded correctly.



### 5\. Run the application



```bash

python app.py

```



Open the local URL displayed in the terminal, typically:



```text

http://127.0.0.1:5000

```



## Model Testing



Run the model test script:



```bash

python test\_model.py

```



The script performs inference on a sample validation image and saves the annotated prediction using Ultralytics.



## Detection Classes



| Class ID | Class   |

|  --- 	   |  ---    |

| 0 	   | Pothole |

| 1 	   | Crack   |

| 2 	   | Patch   |



## Important Limitations



* Detection performance depends on image quality, lighting, and the training dataset.
* Mass, severity, and repair-cost figures are estimates based on configured assumptions; they are not calibrated engineering measurements.
* Reliable real-world measurements would require suitable scale references, camera calibration, and validation against ground-truth road measurements.



## Future Improvements



* Evaluate the model on a diverse Indian road-damage dataset.
* Improve detection performance across different lighting and road conditions.
* Validate damage-size and repair-cost estimation against real measurements.
* Deploy the application to a cloud platform.



## Application Screenshots



### Dashboard



!\[Road Damage Detection Dashboard](Screenshots/Dashboard.png)



### Road Damage Detection Results



!\[YOLOv8 Detection Results](Screenshots/result.png)



### Damage Analysis Summary



!\[Road Damage Analysis Summary](Screenshots/Analysis.png)





## Author



**Mohammed Asif**



GitHub: https://github.com/mdasif1404



\---



*Developed as a deep learning and computer vision project for road damage detection.*

