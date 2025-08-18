# RHEL

## EC2 Requests

For internal Red Hat users, you can request an AWS Blank Open Environment from demo.redhat.com using the following catalog item:
https://catalog.demo.redhat.com/catalog?item=babylon-catalog-prod/sandboxes-gpte.sandbox-open.prod&utm_source=webapp&utm_medium=share-link

Create a new EC2 Instance using the `RHEL` quick start.  

For the instance type, select `g5.4xlarge`

Create a new Key pair login.

For network configuration, create a new security group and select all three options for `Allow SSH traffic from`, `Allow HTTPS traffic from the internet`, and `Allow HTTP traffic from the internet`.

For storage, request `100 GiB`

## Setup Instructions

Register system
Navigate to https://console.redhat.com/insights/registration/
Auto-generate a registration command and run it on the RHEL system
```
sudo rhc connect --activation-key <activation-key> --organization <organization>
```

Install podman
```
sudo dnf -y install podman
podman --version
```

Install CUDA #this may not be working
```
sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
sudo dnf install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r) gcc make dkms acpid libglvnd-glx libglvnd-opengl libglvnd-devel pkgconfig

sudo dnf config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel9/$(uname -i)/cuda-rhel9.repo
sudo dnf module install -y nvidia-driver:open-dkms
nvidia-smi
```

```
sudo dnf install cuda-toolkit
sudo dnf install nvidia-gds
sudo reboot
nvidia-smi
```

Install nvidia-container-toolkit
```
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf-config-manager --enable nvidia-container-toolkit-experimental
sudo dnf install -y nvidia-container-toolkit
```

Configure CDI
```
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# check the config
nvidia-ctk cdi list
```

Validate GPU passthrough is working
```
sudo podman run --rm --device nvidia.com/gpu=all docker.io/nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

## Logging Into Red Hat Container Registry

Create a red hat container registry service account if you don't have one already:

https://access.redhat.com/terms-based-registry/

After creating the SA, find it in the list and open it to access the token.  Copy the `Docker login` command and replace the `docker` command with `sudo podman`.  Execute that in your environment:

```
sudo podman login -u='<sa-name>' -p=<token> registry.redhat.io
```

## Running vLLM

Create vllm pod
```
sudo podman kube play vllm-pod.yaml
```

Remove and cleanup vllm pod
```
sudo podman pod stop vllm && sudo podman pod rm vllm
```

Follow logs
```
sudo podman logs --follow vllm-vllm 
```

```
curl http://localhost/version
# this is accessible from the internet
curl http://<public-ip-address>/version
```
