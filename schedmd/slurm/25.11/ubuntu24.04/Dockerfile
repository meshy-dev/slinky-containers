# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (C) SchedMD LLC.
# SPDX-License-Identifier: Apache-2.0

################################################################################
# Build args
#
# PARENT_IMAGE            Base image for all CUDA/EFA stages (ubuntu:24.04).
# CUDA_TOOLKIT_VERSION    apt suffix for cuda-toolkit packages (e.g. 13-2).
# SLURM_VERSION           Upstream SchedMD slurm source version (e.g. 25.11.5).
# SLURM_DEB_PATCH_REV     Local .deb patch revision stamped into debian/changelog.
# EFA_INSTALLER_VERSION   aws-efa-installer release version.
# EFA_INSTALLER_FLAGS     Flags passed to ./efa_installer.sh. Includes --mpi=openmpi5
#                         so only the OMPI 5 stack (+ libpmix-aws + prrte-aws) is laid
#                         down under /opt/amazon/openmpi5; the OMPI 4 tree at
#                         /opt/amazon/openmpi is never created. Do not pass --skip-mpi
#                         (that would skip OMPI 5 too).
# OMPI_DEFAULT            Which bundled OMPI tree to export on runtime PATH (openmpi5).
# NCCL_CUDA_SUFFIX        apt version suffix for libnccl2/libnccl-dev (e.g. cuda13.2).
# NVCC_GENCODE            NVCC gencode flags passed through to NCCL / plugin builds.
# DEBIAN_FRONTEND         standard apt noninteractive.
################################################################################

ARG PARENT_IMAGE=ubuntu:24.04
ARG CUDA_TOOLKIT_VERSION=13-2
ARG SLURM_VERSION=25.11.5
ARG SLURM_VERSION_MICRO=1
ARG SLURM_DEB_PATCH_REV=meshy1
ARG EFA_INSTALLER_VERSION=1.48.0
ARG EFA_INSTALLER_FLAGS="-y --skip-kmod --skip-limit-conf --no-verify --mpi=openmpi5"
ARG OMPI_DEFAULT=openmpi5
ARG NCCL_CUDA_SUFFIX=cuda13.2
ARG NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"
ARG SLURM_DIR=slurm
ARG FABRIC=ib
ARG FABRIC_BASE=base-${FABRIC}

################################################################################
# slurm-src — alpine sidecar: fetch + extract SchedMD slurm source tarball.
FROM alpine:latest AS slurm-src

ARG SLURM_VERSION
ARG SLURM_VERSION_MICRO
ARG SLURM_DIR
ARG SLURM_TAR="slurm-${SLURM_VERSION//./-}-${SLURM_VERSION_MICRO}.tar.gz"

WORKDIR /workspace/

RUN apk add -q --no-cache tar

ADD https://github.com/SchedMD/slurm/archive/refs/tags/${SLURM_TAR} ${SLURM_TAR}

RUN \
  mkdir -p ${SLURM_DIR} && \
  tar --strip-components=1 -zxvf ${SLURM_TAR} -C ${SLURM_DIR} && \
  rm ${SLURM_TAR}

################################################################################
# base — plain Ubuntu 24.04 + SchedMD build/runtime userland.
#
# NVIDIA_* env vars below are read by nvidia-container-toolkit at `docker run`
# time to inject the host driver libraries (libcuda.so, libnvidia-ml.so.1) into
# the container. Plain ubuntu base does not set these automatically the way
# nvidia/cuda:*-devel images do; they must be declared here so kubelet +
# nvidia-container-runtime populate the mount correctly at slurmd pod start.
FROM ${PARENT_IMAGE} AS base

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

ARG SLURM_VERSION
ENV SLURM_VERSION=${SLURM_VERSION}

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA="cuda>=13.2"

USER root
WORKDIR /tmp/

ARG SLURM_USER=slurm
ARG SLURM_USER_UID=401
ARG SLURM_USER_GID=401

RUN <<EOR
# Remove the default `ubuntu` user (uid 1000) that ships in ubuntu:24.04 so
# it does not collide with cluster-side uid ranges from sssd. userdel --remove
# on a single known entry is deterministic; no conditional loop needed.
set -xeuo pipefail
userdel --remove ubuntu
EOR

RUN <<EOR
# Create SlurmUser.
set -xeuo pipefail
groupadd --system --gid=${SLURM_USER_GID} ${SLURM_USER}
useradd --system --no-log-init --uid=${SLURM_USER_UID} --gid=${SLURM_USER_GID} --shell=/usr/sbin/nologin ${SLURM_USER}
EOR

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
# Baseline apt deps — SchedMD build-deps (cross-checked against upstream
# debian/control Build-Depends), SSL/crypto/json libs, PAM + MariaDB client
# libs for slurm auth/accounting, locale data, and runtime QoL (tini, gosu,
# supervisor). Kept as one layer to share the apt cache mount.
set -xeuo pipefail
apt-get -qq update
apt-get -qq -y upgrade
apt-get -qq -y install --no-install-recommends \
  ca-certificates curl gnupg wget tini gosu \
  supervisor \
  procps iputils-ping ncurses-bin socat mosh \
  vim less sudo unzip git \
  build-essential pkg-config patch \
  python3 python3-pip \
  tar environment-modules \
  locales tzdata \
  libhwloc-dev hwloc \
  libmunge-dev libmunge2 munge \
  libjson-c-dev libyaml-dev liblz4-dev libssl-dev \
  libhttp-parser-dev libjwt-dev libdbus-1-dev \
  libpam0g-dev libmariadb-dev
# Locale+TZ defaults so debuild and slurm logs emit ASCII UTC timestamps
# deterministically across build hosts.
locale-gen en_US.UTF-8
ln -fs /usr/share/zoneinfo/UTC /etc/localtime
EOR

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

################################################################################
# base-cuda — CUDA toolkit + NCCL from NVIDIA network repo (driver-free).
#
# Installing cuda-toolkit-${CUDA_TOOLKIT_VERSION} with --no-install-recommends
# pulls nvcc + runtime libs + the /usr/local/cuda alternatives symlink without
# pulling cuda-drivers (those live under separate meta packages). Runtime
# libnvidia-ml / libcuda come from the host driver at `docker run` time via
# nvidia-container-toolkit keyed on NVIDIA_* env vars set in `base`.
#
# Package suffix pin on libnccl2 / libnccl-dev (e.g. *+cuda13.2) guards against
# NCCL↔CUDA ABI skew: the CUDA network repo ships multiple libnccl2 builds per
# NCCL release, one per CUDA major.minor, and grabbing the wrong one yields
# symbol mismatches against the CUDA runtime.
FROM base AS base-cuda

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_TOOLKIT_VERSION
ARG NCCL_CUDA_SUFFIX

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
# cuda-keyring installs the NVIDIA CUDA apt signing key + repo list under
# /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list.
cd /tmp
curl -fsSL -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
apt-get -qq update
# Driver-free toolkit + NVML dev headers + minimal compiler subset. --no-install-recommends
# is essential: recommended deps include the display driver meta package.
apt-get -qq -y install --no-install-recommends \
  cuda-toolkit-${CUDA_TOOLKIT_VERSION} \
  cuda-nvml-dev-${CUDA_TOOLKIT_VERSION} \
  cuda-minimal-build-${CUDA_TOOLKIT_VERSION} \
  libnccl2 libnccl-dev
test -L /usr/local/cuda
EOR

ENV PATH=/usr/local/cuda/bin:${PATH}

################################################################################
# base-efa — AWS EFA installer: libfabric-aws + OMPI5 + aws-ofi-nccl.
# Fabric: efa
#
# Plain ubuntu:24.04 has no /opt/nvidia/nvidia_entrypoint.sh marker, so the
# EFA installer runs in normal (non-NGC) mode: it installs its own bundled
# rdma-core v61, its OMPI 5 tree (openmpi50-aws + libpmix-aws + prrte-aws)
# under /opt/amazon/openmpi5 and /opt/amazon/pmix, and the non-NGC
# aws-ofi-nccl plugin. The EFA_INSTALLER_FLAGS ARG passes --mpi=openmpi5 so
# the installer selects only the OMPI 5 flavor (valid choices per 1.48.0
# --help: openmpi4,openmpi5,none); the OMPI 4 tree at /opt/amazon/openmpi is
# never laid down. No manual dpkg -i and no package removals are needed —
# dependency resolution still goes through the installer's apt codepath.
#
# OMPI5 + its bundled PMIx v4 are required by slurm-build:
#   --with-pmix=/opt/amazon/pmix in debian-rules-nvml-pmix.patch makes slurmd
#   link the PMIx ABI that user jobs consume at launch via srun --mpi=pmix.
# User jobs then invoke /opt/amazon/openmpi5/bin/mpirun at runtime.
FROM base-cuda AS base-efa

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG EFA_INSTALLER_VERSION
ARG EFA_INSTALLER_FLAGS
ARG OMPI_DEFAULT

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
cd /tmp
curl -fsSL -O "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
tar -xf "aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
cd aws-efa-installer
# shellcheck disable=SC2086
./efa_installer.sh ${EFA_INSTALLER_FLAGS}
cd /tmp
rm -rf aws-efa-installer "aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
EOR

# EFA post-install fail-fast gates. In non-NGC mode the installer drops the
# aws-ofi-nccl plugin at the flat /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so
# path (the non-NGC libnccl-ofi deb uses the same flat layout as the NGC-v3
# variant; the multiarch x86_64-linux-gnu subdir is not used by EFA 1.48.0
# on ubuntu:24.04). The EFA installer only adds /opt/amazon/{efa,ofi-nccl}/lib
# to ld.so.conf.d/ and expects OMPI5 + PMIx to be reached via OPAL_PREFIX and
# LD_LIBRARY_PATH at runtime; register them in ld.so.conf.d so that `ldconfig
# -p` resolves libpmix.so.2 consistently whether or not LD_LIBRARY_PATH is set
# (slurmd's mpi_pmix_v4 plugin dlopen()s libpmix.so.2 at job-step launch).
RUN <<EOR
set -xeuo pipefail
test -x /opt/amazon/openmpi5/bin/mpirun
# AWS-bundled mpirun prints `mpirun (Open MPI) 5.0.9amzn1` — match on the
# close-paren + major to avoid matching other "MPI 5" substrings.
/opt/amazon/${OMPI_DEFAULT}/bin/mpirun --version | grep -E "\(Open MPI\) 5\." >/dev/null \
  || { echo "FATAL: /opt/amazon/${OMPI_DEFAULT}/bin/mpirun did not report Open MPI 5.x" >&2; exit 1; }
test -f /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so
test -f /opt/amazon/pmix/lib/libpmix.so.2
# /opt/amazon/openmpi (OMPI 4 tree) must NOT exist: --mpi=openmpi5 in
# EFA_INSTALLER_FLAGS selects only the OMPI 5 flavor. Catching a stray OMPI 4
# install surfaces upstream flag drift (e.g., someone reverting to default
# flags or appending --mpi=openmpi4) before it ships.
test ! -e /opt/amazon/openmpi \
  || { echo "FATAL: /opt/amazon/openmpi exists; EFA installer laid down OMPI 4 despite --mpi=openmpi5" >&2; exit 1; }
# openmpi40-aws deb must not be installed either. `dpkg -l openmpi40-aws`
# exits nonzero when the package is unknown to dpkg; we invert with `!` so
# the fail-fast || branch fires only if dpkg DID find the package installed.
# Redirect both streams to /dev/null so the negative path stays quiet.
! dpkg -l openmpi40-aws >/dev/null 2>&1 \
  || { echo "FATAL: openmpi40-aws deb is installed; EFA_INSTALLER_FLAGS must pass --mpi=openmpi5 (not default, not --mpi=openmpi4)" >&2; exit 1; }
printf '%s\n' "/opt/amazon/${OMPI_DEFAULT}/lib" "/opt/amazon/pmix/lib" > /etc/ld.so.conf.d/200_amazon-ompi-pmix.conf
ldconfig
ldconfig -p | grep -q 'libpmix\.so\.2 ' || { echo "FATAL: libpmix.so.2 not resolvable via ldconfig after registering /opt/amazon/pmix/lib" >&2; exit 1; }
EOR

################################################################################
# base-ib — DOCA userland IB verbs + Open MPI + system PMIx.
#
# Installs only guest/userland components (no kmod, no metapackage bloat).
# Open MPI comes from DOCA repo (currently 4.1.x). PMIx comes from Ubuntu
# because DOCA repo does not ship PMIx debs.
FROM base-cuda AS base-ib

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
mkdir -p -m 755 /etc/apt/keyrings
curl -fsSL https://linux.mellanox.com/public/repo/doca/latest/ubuntu24.04/x86_64/GPG-KEY-Mellanox.pub | gpg --dearmor -o /etc/apt/keyrings/doca.gpg
chmod 644 /etc/apt/keyrings/doca.gpg
echo "deb [signed-by=/etc/apt/keyrings/doca.gpg arch=amd64] https://linux.mellanox.com/public/repo/doca/latest/ubuntu24.04/x86_64/ /" | tee /etc/apt/sources.list.d/doca.list
chmod 644 /etc/apt/sources.list.d/doca.list
apt-get -qq update
apt-get -qq -y install --no-install-recommends \
  rdma-core \
  ibverbs-providers \
  ibverbs-utils \
  libibverbs1 \
  libibverbs-dev \
  librdmacm1 \
  librdmacm-dev \
  infiniband-diags \
  openmpi \
  libpmix-dev \
  libpmix2t64
# DOCA openmpi installs under /usr/mpi/gcc/openmpi-*/ but does not create
# /usr/bin symlinks. Expose a version-agnostic MPI_HOME and put mpirun on
# PATH so the post-install gate and downstream slurmd stage resolve it.
OMPI_DIR=$(ls -d /usr/mpi/gcc/openmpi-* | head -1)
ln -sfn "${OMPI_DIR}" /usr/mpi/gcc/openmpi
for b in mpirun mpiexec mpicc mpicxx; do
  [ -e "/usr/mpi/gcc/openmpi/bin/${b}" ] && ln -sf "/usr/mpi/gcc/openmpi/bin/${b}" "/usr/bin/${b}" || true
done
ldconfig
EOR

# Post-install verification gates — mirror EFA fail-fast style.
RUN <<EOR
set -xeuo pipefail
test -x /usr/bin/mpirun
/usr/bin/mpirun --version | grep -E "\(Open MPI\) [45]\." >/dev/null \
  || { echo "FATAL: /usr/bin/mpirun did not report Open MPI 4.x/5.x" >&2; exit 1; }
ldconfig
ldconfig -p | grep -q 'libpmix\.so\.2 ' || { echo "FATAL: libpmix.so.2 not resolvable via ldconfig" >&2; exit 1; }
# ibv_devinfo fails when no IB device exists (expected in CI build), so treat as non-fatal.
ibv_devinfo >/dev/null 2>&1 || echo "WARN: ibv_devinfo failed (no RDMA device during build, expected)" >&2
true
EOR

################################################################################
# slurm-build — build SchedMD .debs from source against parameterized fabric
# environment.
#
# Parent is selected via FABRIC_BASE (base-efa or base-ib). Patches are applied
# from patches/common/ then patches/${FABRIC}/ so EFA and IB paths receive the
# correct debian/control and debian/rules hunks (e.g., EFA drops apt libpmix-dev
# and points to /opt/amazon/pmix; IB keeps apt libpmix-dev for system PMIx).
ARG FABRIC=ib
ARG FABRIC_BASE=base-${FABRIC}
FROM ${FABRIC_BASE} AS slurm-build

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG SLURM_VERSION
ARG SLURM_DIR
ARG SLURM_DEB_PATCH_REV
ARG FABRIC=ib
ENV SLURM_VERSION=${SLURM_VERSION}
ENV SLURM_DEB_PATCH_REV=${SLURM_DEB_PATCH_REV}

USER root
WORKDIR /tmp/

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
# Slurm debuild tooling.
set -xeuo pipefail
apt-get -qq -y install --no-install-recommends \
  apt-utils fakeroot devscripts equivs
EOR

COPY --from=slurm-src /workspace/${SLURM_DIR} ${SLURM_DIR}
COPY patches/ /tmp/patches/

RUN <<EOR
# Apply debian/ patches before build-dep resolution so mk-build-deps sees
# the patched debian/control.
set -xeuo pipefail
cd /tmp/${SLURM_DIR}
for p in /tmp/patches/common/*.patch; do
  [ -e "$p" ] || continue
  patch -p1 < "$p"
done
for p in /tmp/patches/${FABRIC}/*.patch; do
  [ -e "$p" ] || continue
  patch -p1 < "$p"
done
EOR

# Ref: https://slurm.schedmd.com/quickstart_admin.html#debuild
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
mk-build-deps -ir --tool='apt-get -y -o Debug::pkgProblemResolver=yes --no-install-recommends' ${SLURM_DIR}/debian/control >/dev/null
( cd ${SLURM_DIR} && debuild -b -uc -us >/dev/null )
EOR

################################################################################
# Legacy build alias — preserves existing non-GPU daemon stages (slurmctld,
# slurmdbd, slurmrestd, sackd, login) that `COPY --from=build`. slurm-build
# replaces it structurally; kept here as a zero-cost alias to avoid breaking
# the other targets in this Dockerfile.
FROM slurm-build AS build

################################################################################
# gdrcopy-build — build gdrcopy userspace (libgdrapi) from source.
# Ref: https://github.com/NVIDIA/gdrcopy
FROM base-cuda AS gdrcopy-build

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

USER root
WORKDIR /workspace/

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
apt-get -qq -y install --no-install-recommends \
  git check debhelper dkms
EOR

RUN <<EOR
set -xeuo pipefail
git clone https://github.com/NVIDIA/gdrcopy.git /workspace/gdrcopy
cd /workspace/gdrcopy
make prefix=/usr/local CUDA=/usr/local/cuda config lib lib_install
EOR

################################################################################
# slurmctld — control-plane daemon (non-GPU; stays on `base`).
FROM base AS slurmctld

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

USER root
WORKDIR /tmp/

COPY --from=build /tmp/*.deb /tmp/*.ddeb /tmp/

# Ref: https://slurm.schedmd.com/quickstart_admin.html#pkg_install
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
apt-get -qq -y install --no-install-recommends --fix-broken \
  socat \
  ./slurm-smd-client_[0-9]*.deb \
  ./slurm-smd-client-dbgsym_[0-9]*.ddeb \
  ./slurm-smd-dev_[0-9]*.deb \
  ./slurm-smd-doc_[0-9]*.deb \
  ./slurm-smd-slurmctld_[0-9]*.deb \
  ./slurm-smd-slurmctld-dbgsym_[0-9]*.ddeb \
  ./slurm-smd_[0-9]*.deb \
  ./slurm-smd-dbgsym_[0-9]*.ddeb
rm *.deb && rm *.ddeb
EOR

COPY files/etc/supervisor/supervisord.conf /etc/supervisor/
COPY \
  files/etc/supervisor/conf.d/slurmctld.conf \
  files/etc/supervisor/conf.d/fakesystemd.conf \
  /etc/supervisor/conf.d/
COPY files/usr/local/bin/fakesystemd.sh /usr/local/bin/
COPY files/usr/local/bin/slurmctld-entrypoint.sh /usr/local/bin/entrypoint.sh

EXPOSE 6817/tcp
ENTRYPOINT ["entrypoint.sh"]

################################################################################
# slurmd — compute daemon with CUDA + EFA + OMPI5 PMIx runtime.
#
# Runtime env:
#   PATH            prepend /opt/amazon/${OMPI_DEFAULT}/bin + /opt/amazon/efa/bin
#   LD_LIBRARY_PATH prepend EFA + OMPI5 + PMIx + ofi-nccl + CUDA
#   OPAL_PREFIX     /opt/amazon/${OMPI_DEFAULT}  (OMPI requires this to find
#                                                 its own prrte/libevent/etc.
#                                                 outside /usr/{lib,share})
#   NCCL_SOCKET_IFNAME=^docker,lo,veth  exclude container virt interfaces
# Deliberately NOT set: FI_EFA_USE_DEVICE_RDMA (default ON for p4/p5/p6 in
# aws-ofi-nccl ≥1.7) and FI_EFA_FORK_SAFE (harmful on Linux ≥5.15 — kernel
# handles MADV_DONTFORK and FORK_SAFE forces COW penalties on PyTorch
# DataLoader / fork-heavy paths).
# Refs: https://github.com/aws/aws-ofi-nccl/blob/master/doc/efa-env-var.md
#
# Parent = slurm-build: direct chain, no IB/OFED auth-scaffold intermediate.
# The forles pod spec runs sshd inside this container (on a non-22 port via
# SSHD_OPTIONS) and mounts authorized_keys from a k8s secret, so only
# openssh-server is installed here — no sssd/authselect/libpam-sss/libnss-sss
# (LDAP/AD integration for on-prem clusters) since auth is munge + the
# k8s-mounted authorized_keys.
FROM slurm-build AS slurmd

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG FABRIC=ib

ARG MPI_HOME=/opt/amazon/openmpi5
ARG EFA_HOME=/opt/amazon/efa
ARG PMIX_HOME=/opt/amazon/pmix
ARG OFI_HOME=/opt/amazon/efa
ARG OPAL_PREFIX=/opt/amazon/openmpi5

USER root
WORKDIR /tmp/

# slurm-build .debs already live on FS via the fabric base -> slurm-build chain;
# install only the subset dpkg consumes here. openssh-server is the only
# non-slurm runtime dependency the forles slurmd pod needs (sshd binds the
# pod's ssh port so operators can `kubectl exec`-free exec into compute nodes).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
apt-get -qq -y install --no-install-recommends --fix-broken \
  openssh-server \
  /tmp/slurm-smd-client_[0-9]*.deb \
  /tmp/slurm-smd-client-dbgsym_[0-9]*.ddeb \
  /tmp/slurm-smd-dev_[0-9]*.deb \
  /tmp/slurm-smd-doc_[0-9]*.deb \
  /tmp/slurm-smd-libnss-slurm_[0-9]*.deb \
  /tmp/slurm-smd-libnss-slurm-dbgsym_[0-9]*.ddeb \
  /tmp/slurm-smd-libpam-slurm-adopt_[0-9]*.deb \
  /tmp/slurm-smd-libpmi2-0_[0-9]*.deb \
  /tmp/slurm-smd-slurmd_[0-9]*.deb \
  /tmp/slurm-smd-slurmd-dbgsym_[0-9]*.ddeb \
  /tmp/slurm-smd_[0-9]*.deb \
  /tmp/slurm-smd-dbgsym_[0-9]*.ddeb
mkdir -p /var/spool/slurmd/
# slurm-smd-libnss-slurm provides a 'slurm' NSS module; prepend it to
# passwd/group lookups so slurm-adopted PAM sessions can resolve job users.
sed -i -E "s/^passwd:[[:space:]]+/&slurm /g" /etc/nsswitch.conf
sed -i -E "s/^group:[[:space:]]+/&slurm /g" /etc/nsswitch.conf
# Remove distro-default host keys; pod-level init regenerates fresh keys
# (or mounts them from a secret) on first boot.
rm -f /etc/ssh/ssh_host_*
EOR

COPY --from=gdrcopy-build /usr/local/lib/libgdrapi* /usr/local/lib/
COPY --from=gdrcopy-build /usr/local/include/gdr* /usr/local/include/
RUN ldconfig

COPY files/etc/supervisor/supervisord.conf /etc/supervisor/
COPY \
  files/etc/supervisor/conf.d/fakesystemd.conf \
  files/etc/supervisor/conf.d/slurmd.conf \
  files/etc/supervisor/conf.d/sshd.conf \
  /etc/supervisor/conf.d/
COPY files/usr/local/bin/fakesystemd.sh /usr/local/bin/
COPY files/usr/local/bin/slurmd-entrypoint.sh /usr/local/bin/entrypoint.sh

ENV PATH=${MPI_HOME}/bin:${EFA_HOME}/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=${EFA_HOME}/lib:${MPI_HOME}/lib:${PMIX_HOME}/lib:${OFI_HOME}/lib:/usr/local/cuda/lib64:/usr/local/lib
ENV OPAL_PREFIX=${OPAL_PREFIX}

# SSH login shells do not automatically inherit Docker ENV declarations;
# ensure profile.d exports match the ENV block above so interactive sessions
# (e.g. `ssh login` or `kubectl exec` into slurmd) resolve mpirun, nvcc, etc.
RUN <<EOR
set -xeuo pipefail
cat > /etc/profile.d/fabric-runtime.sh <<PROFILE
export PATH=${MPI_HOME}/bin:${EFA_HOME}/bin:/usr/local/cuda/bin:\${PATH}
export LD_LIBRARY_PATH=${EFA_HOME}/lib:${MPI_HOME}/lib:${PMIX_HOME}/lib:${OFI_HOME}/lib:/usr/local/cuda/lib64:/usr/local/lib
PROFILE
if [ -n "${OPAL_PREFIX}" ]; then
  echo "export OPAL_PREFIX=${OPAL_PREFIX}" >> /etc/profile.d/fabric-runtime.sh
fi
chmod 644 /etc/profile.d/fabric-runtime.sh
cat > /etc/nccl.conf <<'EOF'
NCCL_SOCKET_IFNAME=^docker,lo,veth
EOF
chmod 644 /etc/nccl.conf
EOR

EXPOSE 6818/tcp 22/tcp
ENTRYPOINT ["entrypoint.sh"]

################################################################################
FROM base AS slurmdbd

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

USER root
WORKDIR /tmp/

COPY --from=build /tmp/*.deb /tmp/*.ddeb /tmp/

# Ref: https://slurm.schedmd.com/quickstart_admin.html#pkg_install
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
apt-get -qq -y install --no-install-recommends --fix-broken \
  ./slurm-smd-doc_[0-9]*.deb \
  ./slurm-smd-slurmdbd_[0-9]*.deb \
  ./slurm-smd-slurmdbd-dbgsym_[0-9]*.ddeb \
  ./slurm-smd_[0-9]*.deb \
  ./slurm-smd-dbgsym_[0-9]*.ddeb
rm *.deb && rm *.ddeb
EOR

EXPOSE 6819/tcp
ENTRYPOINT ["tini", "-g", "--", "slurmdbd", "-D"]

################################################################################
FROM base AS slurmrestd

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

USER root
WORKDIR /tmp/

COPY --from=build /tmp/*.deb /tmp/*.ddeb /tmp/

# Ref: https://slurm.schedmd.com/quickstart_admin.html#pkg_install
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
apt-get -qq -y install --no-install-recommends --fix-broken \
  ./slurm-smd-doc_[0-9]*.deb \
  ./slurm-smd-slurmrestd_[0-9]*.deb \
  ./slurm-smd-slurmrestd-dbgsym_[0-9]*.ddeb \
  ./slurm-smd_[0-9]*.deb \
  ./slurm-smd-dbgsym_[0-9]*.ddeb
rm *.deb && rm *.ddeb
EOR

EXPOSE 6820/tcp
ENTRYPOINT ["tini", "-g", "--", "slurmrestd"]
CMD ["0.0.0.0:6820"]

################################################################################
FROM slurm-build AS sackd

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

USER root
WORKDIR /tmp/

ADD https://dl.k8s.io/release/stable.txt kubernetes_stable.txt

# Ref: https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/#install-using-native-package-management
# Ref: https://slurm.schedmd.com/quickstart_admin.html#pkg_install
# sackd .debs are already on FS via the base-efa -> slurm-build chain.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
VERSION="$(cat kubernetes_stable.txt | grep -Eo '^v[0-9]+\.[0-9]+')"
mkdir -p -m 755 /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/${VERSION}/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/${VERSION}/deb/ /" | tee /etc/apt/sources.list.d/kubernetes.list
chmod 644 /etc/apt/sources.list.d/kubernetes.list
apt-get -qq update
apt-get -qq -y install --no-install-recommends \
  apt-transport-https kubectl
apt-get -qq -y install --no-install-recommends --fix-broken \
  socat \
  /tmp/slurm-smd-client_[0-9]*.deb \
  /tmp/slurm-smd-client-dbgsym_[0-9]*.ddeb \
  /tmp/slurm-smd-doc_[0-9]*.deb \
  /tmp/slurm-smd-sackd_[0-9]*.deb \
  /tmp/slurm-smd-sackd-dbgsym_[0-9]*.ddeb \
  /tmp/slurm-smd_[0-9]*.deb \
  /tmp/slurm-smd-dbgsym_[0-9]*.ddeb
mkdir -p /run/slurm/
chown ${SLURM_USER}:${SLURM_USER} /run/slurm
rm *.txt
EOR

COPY files/etc/supervisor/supervisord.conf /etc/supervisor/
COPY \
  files/etc/supervisor/conf.d/sackd.conf \
  files/etc/supervisor/conf.d/fakesystemd-slurm.conf \
  /etc/supervisor/conf.d/
COPY files/usr/local/bin/fakesystemd.sh /usr/local/bin/
COPY files/usr/local/bin/sackd-entrypoint.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]

################################################################################
# login — researcher SSH entry node. Rebuilt on slurmd base so it inherits
# CUDA + EFA + OMPI5 + PMIx runtime ENV (PATH/LD_LIBRARY_PATH/OPAL_PREFIX).
# All steps from upstream sackd stage are replicated verbatim, followed by all
# steps from upstream login stage verbatim.
# ── login ─────────────────────────────────────────────────────────────────────
FROM slurmd AS login

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

USER root
WORKDIR /tmp/

RUN rm /etc/supervisor/conf.d/slurmd.conf \
    && rm /etc/supervisor/conf.d/fakesystemd.conf

ENV NVIDIA_VISIBLE_DEVICES=void
ENV NVIDIA_DRIVER_CAPABILITIES=
ENV NVIDIA_REQUIRE_CUDA=

# --- replicated from upstream sackd stage (verbatim, minus the FROM line) ---

ADD https://dl.k8s.io/release/stable.txt kubernetes_stable.txt

# Ref: https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/#install-using-native-package-management
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
# Install kubectl
set -xeuo pipefail
apt-get -qq update
VERSION="$(cat kubernetes_stable.txt | grep -Eo '^v[0-9]+\.[0-9]+')"
apt-get -qq -y install --no-install-recommends \
  apt-transport-https ca-certificates curl gnupg
mkdir -p -m 755 /etc/apt/keyrings
curl -fsSL https://pkgs.k8s.io/core:/stable:/${VERSION}/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
chmod 644 /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/${VERSION}/deb/ /" | tee /etc/apt/sources.list.d/kubernetes.list
chmod 644 /etc/apt/sources.list.d/kubernetes.list
apt-get -qq update
apt-get -qq -y install --no-install-recommends kubectl
rm *.txt
EOR

# Ref: https://slurm.schedmd.com/quickstart_admin.html#pkg_install
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
# Install Slurm Packages
set -xeuo pipefail
apt-get -qq update
apt-get -qq -y install --no-install-recommends --fix-broken \
  /tmp/slurm-smd-sackd_[0-9]*.deb \
  /tmp/slurm-smd-sackd-dbgsym_[0-9]*.ddeb
rm *.deb && rm *.ddeb
# Prepare Directories
mkdir -p /run/slurm/
chown ${SLURM_USER}:${SLURM_USER} /run/slurm
EOR

# --- replicated from upstream login stage (verbatim, minus the FROM line) ---

# Ref: https://slurm.schedmd.com/quickstart_admin.html#pkg_install
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
# Install Packages
set -xeuo pipefail
apt-get -qq update
apt-get -qq -y install --no-install-recommends \
  authselect sssd sssd-ad sssd-ldap libpam-sss libnss-sss cmake
# Configure
mkdir -p /etc/authselect
authselect select sssd with-mkhomedir --force
rm -f /etc/ssh/ssh_host_*
EOR

COPY \
  files/etc/supervisor/conf.d/fakesystemd-slurm.conf \
  files/etc/supervisor/conf.d/sackd.conf \
  files/etc/supervisor/conf.d/sshd.conf \
  files/etc/supervisor/conf.d/sssd.conf \
  /etc/supervisor/conf.d/
COPY files/usr/local/bin/login-entrypoint.sh /usr/local/bin/entrypoint.sh

EXPOSE 22/tcp
ENTRYPOINT ["entrypoint.sh"]
