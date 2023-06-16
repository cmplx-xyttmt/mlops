FROM gcr.io/deeplearning-platform-release/pytorch-gpu

WORKDIR /root

COPY requirements.txt .
RUN pip3 install -U -r requirements.txt

COPY . /trainer

WORKDIR /trainer

ENTRYPOINT ["python", "-m", "trainer.task"]

