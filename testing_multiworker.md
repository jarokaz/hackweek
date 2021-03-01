## Simulating multi-worker distributed training with multi-GPU nodes an a single multi-GPU node.


```
docker run --rm -it --gpus '"device=0,1"' \
--env TF_CONFIG='{"cluster": {"worker": ["localhost:12345", "localhost:23456"]}, "task": {"type": "worker", "index": 0} }' \
--network=host \
gcr.io/jk-demos/imdb_bert --epochs=2 --steps_per_epoch=625 --eval_steps=10 --auto_shard_policy=data
```

```
docker run --rm -it --gpus '"device=2,3"' \
--env TF_CONFIG='{"cluster": {"worker": ["localhost:12345", "localhost:23456"]}, "task": {"type": "worker", "index": 1} }' \
--network=host \
gcr.io/jk-demos/imdb_bert --epochs=2 --steps_per_epoch=625 --eval_steps=10 --auto_shard_policy=data
```