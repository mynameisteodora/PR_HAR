## Project Structure

The code assumes the following project structure:

Project

 |
 
 +-- Data
 
 |  |
 
 |  +-- subject_name_1
 |  +-- subject_name_2
 |     |
 |     +-- correct
 |        |
 |        +-- subject_activity_correctness_timestamp.csv
 |        +-- subject_activity_correctness_timestamp.csv
 |     +-- incorrect
 |        |
 |        +-- activity_1.csv
 |        +-- activity_2.csv
 |
 +-- Notebooks
 |  |  
 |  +-- 00_notebook.ipynb
 |  +-- 01_notebook.ipynb
 |
 +-- Plots
 |
 +-- utils (package)
 |  |  
 |  +-- constants.py
 |  +-- file_readers.py
 |  +-- signal_processing.py
 
### Data
The data is organised by subject ID, and then by correctness. 
All files should be in csv format and the header is always 7 rows (recorded on the old Respeck V4 app).

The name of each recording file should follow the format:
"subjectName_activityName_correctness_timestamp.csv"

The data is recorded at 25Hz and contains both accelerometer and gyroscope data. 
It is recorded using the Respeck V4 and should be **downsampled** to 12.5Hz before using. 
We **do not** use the gyroscope data in this project, but keep it in the files for future research.

### Notebooks
Here is where all the Jupyter Notebooks live. Note that if you wish to import any code from the project packages, 
you need to add these two lines at the top of every notebook:
`import sys`
`sys.path.append(..)`

### Utils
Here is where all the utility functions exist. The most important file is file_readers.py, where all functions for 
reading and preprocessing signals live. 
