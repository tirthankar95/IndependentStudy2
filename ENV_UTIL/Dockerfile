FROM dokken/ubuntu-20.04
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install seaborn
RUN pip3 install torch

RUN cd lb-foraging
RUN pip3 install -e . 

RUN cd ../robotic-warehouse
RUN pip3 install -e . 

RUN cd ..
