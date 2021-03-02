
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-4

RUN pip install pip install tf-models-official tensorflow-text 

WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]
