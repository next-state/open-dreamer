# Multinode Dynamics on Kubernetes

This note explains how [`k8s/train-dynamics-multinode-statefulset.yaml`](/Users/francescosacco/github/dreamer4-jax-private/k8s/train-dynamics-multinode-statefulset.yaml) works and how to refresh it safely.

## What the manifest does

The manifest creates:

- A headless `Service` named `train-dynamics-multinode`
- A `StatefulSet` named `train-dynamics-multinode`

Each pod runs one JAX process. Pod `0` is the coordinator and the other pods join it through:

```bash
JAX_COORDINATOR_ADDRESS=train-dynamics-multinode-0.train-dynamics-multinode:12345
JAX_PROCESS_COUNT=14
JAX_PROCESS_INDEX=${HOSTNAME##*-}
```

The headless service is required so each pod gets a stable DNS name like:

```text
train-dynamics-multinode-0.train-dynamics-multinode
```

## How code gets into the pods

The pods do not mount the git checkout directly. Instead:

1. A local tarball is created
2. The tarball is stored in `ConfigMap/dreamer4-src`
3. The `unpack-source` init container extracts it into `/workspace`
4. The main container runs `uv sync`, `uv pip install -e .`, and then launches training

That means changing local files is not enough. After any code or dependency change, you must rebuild the tarball and recreate the ConfigMap.

Recommended bundle command:

```bash
cd /Users/francescosacco/github/dreamer4-jax-private
tar -czf /tmp/dreamer4-k8s-src.tgz \
  dreamer \
  scripts \
  configs \
  pyproject.toml \
  uv.lock
kubectl delete configmap dreamer4-src --ignore-not-found
kubectl create configmap dreamer4-src \
  --from-file=dreamer4-k8s-src.tgz=/tmp/dreamer4-k8s-src.tgz
```

## Required cluster objects

The manifest expects these objects to already exist:

- `ConfigMap/dreamer4-src`
- `Secret/wandb-credentials`

Example secret creation:

```bash
kubectl create secret generic wandb-credentials \
  --from-literal=WANDB_API_KEY="$WANDB_API_KEY"
```

## Running the job

Apply the manifest:

```bash
kubectl apply -f k8s/train-dynamics-multinode-statefulset.yaml
```

Watch status:

```bash
kubectl get statefulset train-dynamics-multinode
kubectl get pods -l app=train-dynamics-multinode -o wide
kubectl logs -f train-dynamics-multinode-0 -c trainer
```

Delete it:

```bash
kubectl delete -f k8s/train-dynamics-multinode-statefulset.yaml
```

## Why ARM nodes broke dependency install

This workload was scheduled onto `Linux aarch64` nodes. Some dependencies in this repo, notably `decord` and `procgen-mirror`, only ship Linux wheels for `x86_64`.

For latent dynamics training, those packages are not needed at runtime, but they were still being pulled in by the base dependency set. The fix is:

- Guard platform-specific dependencies in `pyproject.toml`
- Regenerate `uv.lock`
- Rebuild the source tarball
- Recreate `ConfigMap/dreamer4-src`
- Restart or recreate the pods

If the ConfigMap is not refreshed, Kubernetes will keep running the old source bundle even if the local repo is fixed.

## Operational notes

- `hostPath: /data` means every selected node must have the dataset mounted at `/data`
- The pod anti-affinity rule tries to spread pods across distinct hosts
- `podManagementPolicy: Parallel` allows fast startup, but failures will also fan out quickly
- A `CrashLoopBackOff` here usually means the container command failed, not that scheduling failed
