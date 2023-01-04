# sync env vars
source ~/.bashrc
# install normal python deps
pip install -r requirements.txt
# clone pyhive utilities lance created
git clone git@github.com:ContextLogic/pyhive_utilities.git $HOME/pyhive_utilities
# install python deps for pyhive
pip install -r $HOME/pyhive_utilities/requirements.txt
pip install -U s3fs # fix dvc s3 dependency
# install chromium and its driver for web scraping
sudo apt-get update && sudo apt-get install -y chromium chromium-driver