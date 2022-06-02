#FROM conda/miniconda3
FROM continuumio/miniconda3

RUN apt-get update -y

ENV AZURE_STORAGE_CONNECTION_STRING "DefaultEndpointsProtocol=https;AccountName=mlflowstorageaai;AccountKey=Ga8sxz8dCbexGeKjEEHnaRszOzEXULF0YCBDv6QGaEHKU6xW3Qsy7s+1iDfTofjbVnFzzQfAVfbu+AStI9qrrQ==;EndpointSuffix=core.windows.net"
ENV MLFLOW_SERVER_DEFAULT_ARTIFACT_ROOT "wasbs://mlflowcontainer@mlflowstorageaai.blob.core.windows.net"
ENV AZURE_STORAGE_ACCESS_KEY "XVxhw8bLUPtCDx6+YcmzLoXh4l2iPEdsXrCJq9WQL+LBsbXvWxZutOTouOoBVfXInSjoRC2UPISu+ASt7/ouIw=="


WORKDIR /

COPY . /

RUN conda env create -f /environment_droplet.yml
#RUN conda init
RUN conda init bash
RUN echo "conda activate mlflow-env" > ~/.bashrc
# RUN /root/.bashrc
RUN apt-get update -y
# RUN conda activate mlflow-env
# CMD python3 /ml_flow_song_copy.py
CMD ["conda", "run", "--no-capture-output", "-n", "mlflow-env", "python3", "/ml_flow_song_copy.py"]
