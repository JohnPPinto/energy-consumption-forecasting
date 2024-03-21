FROM apache/airflow:2.8.3-python3.11

USER root

RUN apt-get -y update && ACCEPT_EULA=Y apt-get -y upgrade
RUN apt-get install -y build-essential

USER airflow