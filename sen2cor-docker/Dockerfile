FROM ubuntu:18.04

RUN apt-get update && \
apt-get -y install sudo && \
sudo apt-get install -y software-properties-common

RUN sudo add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable && \
sudo apt-get update && \
sudo apt-get install -y gdal-bin && \
sudo apt-get install -y libgdal-dev && \ 
sudo apt-get update && \
sudo apt-get -yf upgrade

RUN apt-get install wget

RUN wget --quiet http://step.esa.int/thirdparties/sen2cor/2.5.5/Sen2Cor-02.05.05-Linux64.run && \
chmod +x Sen2Cor-02.05.05-Linux64.run && \
./Sen2Cor-02.05.05-Linux64.run

RUN echo "source /Sen2Cor-02.05.05-Linux64/L2A_Bashrc" >> ~/.bashrc

RUN sed -i -e '8s/1/AUTO/' ./root/sen2cor/2.5/cfg/L2A_GIPP.xml && \
sed -i -e '12s/NONE/dem\/srtm/' ./root/sen2cor/2.5/cfg/L2A_GIPP.xml
