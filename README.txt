NEURODRIVE AI
Installation and Running Instructions

*************************************************************************************************************************************************************************************************************************************************************
1. PROJECT OVERVIEW

NeuroDrive AI is a driver-monitoring prototype designed to identify possible signs of drowsiness and distraction using a webcam. It analyses facial landmarks, eye closure, mouth movement, and head orientation. The application provides visual status updates, audio warnings on supported Windows systems, and session telemetry.

*************************************************************************************************************************************************************************************************************************************************************
2. PROJECT FILES

Keep the following files and folders together in your project directory:

project/
    main.py
    evaluate.py
    face_landmarker.task
    README.txt
    dataset/
        drowsy/
        notdrowsy/

Purpose of each file:

main.py: Runs the NeuroDrive AI Streamlit application and live webcam monitoring.

evaluate.py: Evaluates the detection pipeline using labelled dataset images and reports performance metrics.

face_landmarker.task: Pre-trained MediaPipe Face Landmarker model asset required by the application and evaluation script.

README.txt: Contains project information, installation instructions, and usage details.

dataset/drowsy/: Contains images labelled as drowsy for evaluation.

dataset/notdrowsy/: Contains images labelled as not drowsy for evaluation.

The dataset folders are required only for dataset evaluation, not for the normal live webcam demonstration.
*************************************************************************************************************************************************************************************************************************************************************

3. SYSTEM REQUIREMENTS

- Windows computer
- Python installed
- Webcam

The application processes webcam frames locally and is designed to operate without a cloud-based detection service.
*************************************************************************************************************************************************************************************************************************************************************

4. INSTALLATION

Open Command Prompt and navigate to the project folder. If the folder is on your Desktop and is named project, run:

cd /d "%USERPROFILE%\Desktop\project"

Replace project with the actual folder name or change the path if the folder is stored elsewhere.

Check that the required files are present by running:

dir

Install the required Python libraries using these commands:

python -m pip install --upgrade pip

python -m pip install streamlit opencv-python mediapipe numpy pandas matplotlib scikit-learn

These commands install the packages for the active Python environment. Compatibility depends on the Python version and the availability of compatible package releases. If installation fails, use a Python version supported by all required libraries.
*************************************************************************************************************************************************************************************************************************************************************
5. RUNNING THE MAIN APPLICATION

From the project directory, run:

python -m streamlit run main.py

Wait for Streamlit to start. The application should open in your default browser. If it does not, copy the local URL displayed in Command Prompt, usually:

http://localhost:8501

Allow camera access if prompted.

Follow the on-screen calibration instructions by looking straight at the webcam and holding your head steady. After calibration, the dashboard displays monitoring status and available measurements.

Keep the Command Prompt window open while using the application. To stop the Streamlit server, return to Command Prompt and press Ctrl + C.
*************************************************************************************************************************************************************************************************************************************************************

6. OPTIONAL: DATASET EVALUATION

Dataset evaluation is optional and is separate from the live webcam demonstration.

Organise the dataset as follows:

dataset/
    drowsy/
        image files
    notdrowsy/
        image files

Use the exact folder names shown above. Place each image in the appropriate labelled folder. Supported image formats depend on the implementation of evaluate.py. Common formats include JPG, JPEG, PNG, and BMP if supported by the script.

Run the evaluation script from the project directory:

python evaluate.py

The script processes the available evaluation samples and reports metrics such as accuracy, precision, recall, F1-score, and the confusion matrix. False-positive and false-negative rates may also be reported if implemented in the script.

When evaluation finishes, check the evaluation_results folder for generated reports. The exact output files depend on the version of evaluate.py.

Evaluation time depends on the number of images and the computer's processing speed. The reported metrics describe the evaluated samples and should not be interpreted as a guarantee of real-world driving performance.
*************************************************************************************************************************************************************************************************************************************************************
7. DATASET SOURCE AND ATTRIBUTION

Dataset title: Driver Drowsiness Dataset (DDD)

Source: Kaggle

URL: https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd?resource=download-directory

The dataset is used for offline evaluation.
*************************************************************************************************************************************************************************************************************************************************************

8. DEPENDENCIES AND LIBRARIES

Streamlit: Used to create the web-based dashboard and user interface.

OpenCV (opencv-python): Used for webcam capture and image processing.

MediaPipe: Used for facial landmark detection.

NumPy: Used for numerical calculations and facial landmark measurements.

Pandas: Used for telemetry, event logs, and CSV exports.

Matplotlib: Used to create graphs of recorded measurements.

Scikit-learn: Used for evaluation metrics if imported by the evaluation script.

Python standard library: Used for timing, file handling, timestamps, threading, and Windows audio alerts where supported.
*************************************************************************************************************************************************************************************************************************************************************


9. AI TOOLS AND MODEL DISCLOSURE

NeuroDrive AI uses the MediaPipe Face Landmarker model asset, face_landmarker.task, for facial landmark detection. The project applies facial landmark measurements and programmed decision rules to estimate possible drowsiness and distraction.

OpenAI ChatGPT was used as an AI-assisted development and debugging tool to help refine code, troubleshoot errors, and improve the application interface. The implementation was reviewed and adapted for this project.

MediaPipe provides the pre-trained facial landmark model. 

