# Using GCP GPUs

## Setting up the experimentation environment

### Setting up user credentials

From the AI Platform Notebooks instance terminal:

```
gcloud auth login
```
### Provisioning AI Platform Notebooks instance

TBD

### Installing AI Platform (Unified) SDK

From the AI Platform Notebooks instance terminal:

```
pip install -U google-cloud-aiplatform --user
```


### Install Model Garden

```
pip install tf-models-official tensorflow-text
```


### Cloning TF Model Garden repo

```
cd 

export MODELS_BRANCH=master

git clone -b $MODELS_BRANCH  --single-branch https://github.com/tensorflow/models.git

```

### Installin Model Garden dependencies

```
cd models
pip install --user -r official/requirements.txt
```
