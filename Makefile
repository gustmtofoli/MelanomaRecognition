install-python:
	sudo apt-get update
	sudo apt-get upgrade
	sudo add-apt-repository ppa:deadsnakes/ppa
	sudo apt-get update
	sudo apt-get install python3.6

install-virtualenv:
	wget https://bootstrap.pypa.io/get-pip.py
	sudo python3.6 get-pip.py
	sudo pip3 install -U pip
	python3 -m venv virtualenv --without-pip
	
install-tensorflow:
	sudo pip3 install -U tensorflow

install-dependencies:
	sudo pip3 install opencv-python
	sudo apt-get python3-dev

# sciekit-learn
