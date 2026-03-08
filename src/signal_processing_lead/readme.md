Signal Processing Pipeline:

1.    Raw dataset
          ↓
2.    Data loading
          ↓
3.    Preprocessing
          ↓
4.    Sampling / Resampling
          ↓
5.    Filtering / Noise removal
          ↓
6.    Segment selection
          ↓
7.    PSD / frequency analysis
          ↓
8.    Clean signal ready for modulation

to acvtivate environment:

cd /d D:\Telemetry_PBL_Group3
.venv\Scripts\activate

To leave the environment:

deactivate


cd /d D:\Telemetry_PBL_Group3
py --version
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy scipy pandas matplotlib openpyxl jupyter black pylint