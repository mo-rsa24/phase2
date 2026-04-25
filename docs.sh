micromamba create -n phase2-repr python=3.11 -y
micromamba activate phase2-repr
# If requirements.txt contains the Python dependencies:
pip install -r requirements.txt
# If you add packages manually during setup, immediately record them:
pip freeze > requirements.txt
# 4.3 Create a reproducible environment.yml
micromamba env export -n phase2-repr > environment.yml
#4.4 Recreate the environment on another machine
# On the second machine:
micromamba env create -f environment.yml
micromamba activate phase2-repr