# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (C) SchedMD LLC.
# SPDX-License-Identifier: Apache-2.0

################################################################################
# Build args — contract per plan docs/plans/2026-04-19T233208Z-meshy-slinky-slurmd-efa-cuda13.md §3
#
# CUDA_BASE_TAG          NVIDIA CUDA devel image tag (consumed via PARENT_IMAGE)
# SLURM_VERSION          Upstream SchedMD slurm source version (e.g. 25.11.5)
# SLURM_DEB_PATCH_REV    Local .deb patch revision stamped into debian/changelog
# EFA_INSTALLER_VERSION  aws-efa-installer release version
# EFA_INSTALLER_FLAGS    Flags passed to ./efa_installer.sh (MUST NOT include --skip-mpi)
# OMPI_DEFAULT           Which bundled OMPI tree to export on runtime PATH (openmpi5)
# NCCL_CUDA_SUFFIX       apt version suffix for libnccl2/libnccl-dev (e.g. cuda13.2)
# NVCC_GENCODE           NVCC gencode flags passed through to NCCL / plugin builds
# DEBIAN_FRONTEND        standard apt noninteractive
################################################################################

ARG CUDA_BASE_TAG=13.2.1-devel-ubuntu24.04
ARG SLURM_VERSION=25.11.5
ARG SLURM_VERSION_MICRO=1
ARG SLURM_DEB_PATCH_REV=meshy1
ARG EFA_INSTALLER_VERSION=1.48.0
ARG EFA_INSTALLER_FLAGS="-y -g -d --skip-kmod --skip-limit-conf --no-verify"
ARG OMPI_DEFAULT=openmpi5
ARG NCCL_CUDA_SUFFIX=cuda13.2
ARG NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"
ARG SLURM_DIR=slurm
ARG PARENT_IMAGE=nvidia/cuda:${CUDA_BASE_TAG}

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
# base — CUDA devel parent + SchedMD baseline userland.
FROM ${PARENT_IMAGE} AS base

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

ARG SLURM_VERSION
ENV SLURM_VERSION=${SLURM_VERSION}

USER root
WORKDIR /tmp/

ARG SLURM_USER=slurm
ARG SLURM_USER_UID=401
ARG SLURM_USER_GID=401

RUN <<EOR
# Create SlurmUser
set -xeuo pipefail
groupadd --system --gid=${SLURM_USER_GID} ${SLURM_USER}
useradd --system --no-log-init --uid=${SLURM_USER_UID} --gid=${SLURM_USER_GID} --shell=/usr/sbin/nologin ${SLURM_USER}
EOR

RUN <<EOR
# Remove pre-created users from parent CUDA image
set -xeuo pipefail
for user in $(ls /home/); do
  userdel --remove $user || true
done
EOR

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
# Baseline apt deps — init, debug, QoL
set -xeuo pipefail
apt-get -qq update
apt-get -qq -y upgrade
apt-get -qq -y install --no-install-recommends \
  ca-certificates curl gnupg wget tini gosu \
  supervisor \
  procps iputils-ping ncurses-bin \
  vim less sudo unzip git \
  build-essential pkg-config \
  libhwloc-dev libmunge-dev libmunge2 munge
EOR

################################################################################
# base-cuda — NCCL apt pinned to ${NCCL_CUDA_SUFFIX}.
#
# Rationale: install NCCL BEFORE base-efa so EFA installer can see an existing
# NCCL and link the ofi-nccl plugin consistently. Plan §2 / outline §2 DAG:
# base → base-cuda → base-efa. Outline §14 Q2 notes EFA installer may purge
# libnccl2; if B1 exposes that failure mode, swap DAG per outline guidance.
#
# Package suffix pin (e.g. libnccl2=*+cuda13.2) per outline §3 and research
# doc §5 (NCCL↔CUDA skew). Pre-flight apt probe is B4 (outline §14 Q1);
# source-build fallback is B5 (conditional).
FROM base AS base-cuda

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG NCCL_CUDA_SUFFIX

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
# Install NCCL from the CUDA network repo shipped in the parent image.
set -xeuo pipefail
apt-get -qq update
# Candidate selection: newest libnccl2/libnccl-dev whose version contains the
# CUDA suffix token (e.g. "+cuda13.2"). apt-cache policy emits candidate lines;
# if no candidate contains the suffix, this command fails fast (caught by B4).
NCCL_PKG_VER="$(apt-cache madison libnccl2 | awk -v s="${NCCL_CUDA_SUFFIX}" '$3 ~ s { print $3; exit }')"
if [ -z "${NCCL_PKG_VER}" ]; then
  echo "FATAL: no libnccl2 candidate matches suffix ${NCCL_CUDA_SUFFIX}" >&2
  echo "Run B4 pre-flight probe before B1; see plan packet B4." >&2
  exit 1
fi
apt-get -qq -y install --no-install-recommends \
  "libnccl2=${NCCL_PKG_VER}" \
  "libnccl-dev=${NCCL_PKG_VER}"
EOR

################################################################################
# base-efa — AWS EFA installer: libfabric-aws + OMPI4 + OMPI5 + aws-ofi-nccl.
#
# Plan §6 ordering (mirrors research doc §4 pitfalls + §1 canonical install):
#   1. purge conflicting ibverbs / libfabric / openmpi from base image
#   2. apt install EFA build deps (ca-certificates curl tar hwloc etc.)
#   3. fetch aws-efa-installer tarball
#   4. run ./efa_installer.sh ${EFA_INSTALLER_FLAGS}  (NO --skip-mpi)
#   5. verify bundled plugin .so is present
#   6. verify mpirun is OMPI 5.x
#   7. cleanup tarball
#
# EFA_INSTALLER_FLAGS default = "-y -g -d --skip-kmod --skip-limit-conf --no-verify"
# (research doc §1 canonical form). Do NOT append --skip-mpi here or --mpi=openmpi4
# — we NEED the OMPI5 tree for PMIx v4 used by --with-pmix= in slurm-build.
FROM base-cuda AS base-efa

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG EFA_INSTALLER_VERSION
ARG EFA_INSTALLER_FLAGS
ARG OMPI_DEFAULT

# Step 1: purge conflicting system packages that would leak into EFA plugin link.
# Research doc §5 "EFA installer libfabric override" + canonical AWSome-distrib
# `apt-get remove --allow-change-held-packages` stanza.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
apt-get -qq update
apt-get -qq -y remove --allow-change-held-packages \
  libibverbs-dev libmlx5-1 ibverbs-providers libfabric1 libfabric-dev \
  openmpi-bin openmpi-common libopenmpi-dev libopenmpi3 || true
apt-get -qq -y autoremove
EOR

# Step 2-4: fetch and run EFA installer (container-safe flags, with OMPI5).
#
# rdma-core handling in a CUDA devel container:
#  - nvidia/cuda:*-devel-ubuntu24.04 has /opt/nvidia/nvidia_entrypoint.sh,
#    which causes efa_installer.sh to auto-detect the "NGC build image" case
#    (see efa_installer.sh NGC detection path, ~lines 1320-1420). In that
#    mode the installer *skips* installing its own bundled rdma-core v61
#    debs and assumes the base image already provides rdma-core.
#  - Ubuntu 24.04 apt ships rdma-core v50, but EFA 1.48's libfabric1-aws
#    deb has `Depends: ibverbs-providers (>= 59)`, so dpkg -i of
#    libfabric1-aws fails with unsatisfied deps against apt v50.
#
# Fix: pre-install the EFA-bundled rdma-core v61 debs with dpkg before
# efa_installer.sh runs. We install the bundle locally from the extracted
# tarball to avoid an extra download and to guarantee version alignment
# with the EFA release being installed. libnl-* are apt deps of those
# rdma-core v61 debs and have no conflicting bundled version.
#
# environment-modules + curl + tar are installer prerequisites. ca-certs
# must be present before curl fetches the tarball.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
apt-get -qq update
# Prerequisites: installer itself + apt-only runtime deps of bundled
# OMPI5 / PMIx / prrte debs (libevent-*, libhwloc15, zlib1g, libnl-*).
apt-get -qq -y install --no-install-recommends \
  ca-certificates curl tar environment-modules \
  libnl-3-200 libnl-route-3-200 \
  libevent-core-2.1-7t64 libevent-pthreads-2.1-7t64 \
  libhwloc15 zlib1g
cd /tmp
curl -fsSL -O "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
tar -xf "aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
# Pre-install bundled rdma-core v61 runtime debs so libfabric1-aws's
# Depends: ibverbs-providers (>= 59) can be satisfied by dpkg -i. We
# install only the runtime shared libs (not -dev headers or diag tools)
# to keep the image minimal.
cd aws-efa-installer/DEBS/UBUNTU2404/x86_64/rdma-core
dpkg -i \
  libibverbs1_*.deb \
  ibverbs-providers_*.deb \
  librdmacm1_*.deb
cd /tmp/aws-efa-installer
# shellcheck disable=SC2086
./efa_installer.sh ${EFA_INSTALLER_FLAGS}

# EFA installer in NGC-build-image mode only installs libfabric +
# ofi-nccl plugin; it deliberately skips MPI packages (expects the
# NGC parent image to provide them). We need OMPI5 + its PMIx tree
# for two reasons:
#   1. slurm-build stage configures with --with-pmix=/opt/amazon/openmpi5
#      (debian-rules-nvml-pmix.patch) so slurmd links the same PMIx ABI
#      that user jobs will use via srun --mpi=pmix.
#   2. user jobs invoke /opt/amazon/openmpi5/bin/mpirun at runtime.
# Install the three bundled debs (libpmix-aws -> prrte-aws -> openmpi50-aws)
# in dependency order using dpkg -i on the local bundle to guarantee
# version alignment with the EFA release.
cd /tmp/aws-efa-installer/DEBS/UBUNTU2404/x86_64
dpkg -i \
  libpmix-aws_*.deb \
  prrte-aws_*.deb \
  openmpi50-aws_*.deb

cd /tmp
rm -rf aws-efa-installer "aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz"
EOR

# Step 5-6: verify bundled ofi-nccl plugin + OMPI5 mpirun.
# Plugin path in NGC-build-image mode: EFA installer installs the
# libnccl-ofi-ngc-v3 deb which drops the plugin at
# /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so (flat layout, no
# x86_64-linux-gnu subdir). On non-NGC installs EFA instead installs
# libnccl-ofi which uses the x86_64-linux-gnu subdir layout. Accept
# either by listing both candidate paths, but require at least one.
RUN <<EOR
set -xeuo pipefail
# Plugin path (bundled by EFA installer >= 1.35 per research doc §2):
if ls /opt/amazon/ofi-nccl/lib/libnccl-net-ofi*.so >/dev/null 2>&1; then
  ls -l /opt/amazon/ofi-nccl/lib/libnccl-net-ofi*.so
elif ls /opt/amazon/ofi-nccl/lib/*/libnccl-net-ofi*.so >/dev/null 2>&1; then
  ls -l /opt/amazon/ofi-nccl/lib/*/libnccl-net-ofi*.so
else
  echo "Error: no libnccl-net-ofi*.so found under /opt/amazon/ofi-nccl/lib" >&2
  find /opt/amazon/ofi-nccl -name 'libnccl-*' >&2 || true
  exit 1
fi
# OMPI5 mpirun sanity — must report "Open MPI 5."
/opt/amazon/${OMPI_DEFAULT}/bin/mpirun --version | grep -E "Open MPI 5\." >/dev/null
EOR

################################################################################
# slurm-build — build SchedMD .debs from source against base-efa environment.
#
# Parent MUST be base-efa (NOT slurm-src): debian/rules configure flags need
# libpmix from /opt/amazon/openmpi5 and libnvidia-ml-dev from CUDA parent.
# Outline §2 cite: "FROM base-efa (NOT slurm-src — needs OMPI5 libpmix)".
#
# Patches (A3) applied to unpacked source tree before debuild:
#   - debian/control:    adds libnvidia-ml-dev Build-Depends
#   - debian/rules:      adds --with-nvml + --with-pmix=/opt/amazon/openmpi5
#   - debian/changelog:  new entry ${SLURM_VERSION}-1${SLURM_DEB_PATCH_REV}
FROM base-efa AS slurm-build

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG SLURM_VERSION
ARG SLURM_DIR
ARG SLURM_DEB_PATCH_REV
ENV SLURM_VERSION=${SLURM_VERSION}
ENV SLURM_DEB_PATCH_REV=${SLURM_DEB_PATCH_REV}

USER root
WORKDIR /tmp/

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
# Slurm debuild tooling.
set -xeuo pipefail
apt-get -qq update
apt-get -qq -y install --no-install-recommends \
  build-essential apt-utils fakeroot devscripts equivs patch \
  libnvidia-ml-dev
EOR

COPY --from=slurm-src /workspace/${SLURM_DIR} ${SLURM_DIR}
COPY patches/ /tmp/patches/

RUN <<EOR
# Apply SchedMD debian/ patches (A3) before build-dep resolution.
set -xeuo pipefail
cd /tmp/${SLURM_DIR}
for p in /tmp/patches/debian-control-nvml.patch \
         /tmp/patches/debian-rules-nvml-pmix.patch \
         /tmp/patches/debian-changelog-meshy.patch; do
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
# base-extra — Slinky extras MINUS openmpi apt (OMPI comes from /opt/amazon/openmpi5).
#
# Plan §2 / outline §2: "Slinky extras MINUS openmpi-bin/openmpi-common/
# libopenmpi-dev; KEEP libpmix2-equivalent (provided via /opt/amazon/openmpi5);
# add gawk socat authselect sssd openssh-server" (gawk/socat/authselect/sssd/
# ssh moved to slurmd stage where they are used).
FROM base-efa AS base-extra

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
# Mellanox OFED userspace repo (guest IB components only; kernel drivers from host).
set -xeuo pipefail
apt-get -qq update
apt-get -qq -y install --no-install-recommends \
  ca-certificates curl gnupg
curl -fsSL https://linux.mellanox.com/public/repo/mlnx_ofed/latest/ubuntu24.04/mellanox_mlnx_ofed.list | tee /etc/apt/sources.list.d/mellanox-mlnx-ofed.list
wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add -
chmod 644 /etc/apt/sources.list.d/mellanox-mlnx-ofed.list
apt-get -qq update
EOR

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
apt-get -qq update
# NOTE: do NOT install openmpi / openmpi-bin / libopenmpi-dev here — OMPI comes
# from EFA installer at /opt/amazon/openmpi5. Adding apt openmpi re-introduces
# a second OMPI tree and breaks OPAL_PREFIX resolution.
# Install IB userspace guest components; rdma-core already partially pulled by EFA.
apt-get -qq -y install --no-install-recommends \
  rdma-core \
  infiniband-diags \
  mlnx-ethtool \
  perftest
EOR

################################################################################
# gdrcopy-build — build gdrcopy userspace (libgdrapi) from source.
# Ref: https://github.com/NVIDIA/gdrcopy
FROM ${PARENT_IMAGE} AS gdrcopy-build

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

USER root
WORKDIR /workspace/

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
apt-get -qq update
apt-get -qq -y install --no-install-recommends \
  build-essential git check pkg-config debhelper dkms python3
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
apt-get -qq update
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
# Runtime env per plan §6 / outline §6:
#   PATH            prepend /opt/amazon/${OMPI_DEFAULT}/bin + /opt/amazon/efa/bin
#   LD_LIBRARY_PATH prepend EFA + OMPI5 + ofi-nccl + CUDA
#   OPAL_PREFIX     /opt/amazon/${OMPI_DEFAULT}  (research doc line 415)
#   FI_EFA_USE_DEVICE_RDMA=1        (p4/p5/p6 GPUDirect RDMA)
#   FI_EFA_FORK_SAFE=1              (if app forks)
#   NCCL_SOCKET_IFNAME=^docker,lo,veth  (exclude container virt interfaces)
FROM base-extra AS slurmd

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG OMPI_DEFAULT

USER root
WORKDIR /tmp/

COPY --from=slurm-build /tmp/*.deb /tmp/*.ddeb /tmp/

# Ref: https://slurm.schedmd.com/quickstart_admin.html#pkg_install
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
apt-get -qq update
apt-get -qq -y install --no-install-recommends --fix-broken \
  gawk socat \
  openssh-server \
  authselect sssd sssd-ad sssd-ldap libpam-sss libnss-sss \
  ./slurm-smd-client_[0-9]*.deb \
  ./slurm-smd-client-dbgsym_[0-9]*.ddeb \
  ./slurm-smd-dev_[0-9]*.deb \
  ./slurm-smd-doc_[0-9]*.deb \
  ./slurm-smd-libnss-slurm_[0-9]*.deb \
  ./slurm-smd-libnss-slurm-dbgsym_[0-9]*.ddeb \
  ./slurm-smd-libpam-slurm-adopt_[0-9]*.deb \
  ./slurm-smd-libpmi2-0_[0-9]*.deb \
  ./slurm-smd-slurmd_[0-9]*.deb \
  ./slurm-smd-slurmd-dbgsym_[0-9]*.ddeb \
  ./slurm-smd_[0-9]*.deb \
  ./slurm-smd-dbgsym_[0-9]*.ddeb
rm *.deb && rm *.ddeb
mkdir -p /var/spool/slurmd/
sed -i -E "s/^passwd:[[:space:]]+/&slurm /g" /etc/nsswitch.conf
sed -i -E "s/^group:[[:space:]]+/&slurm /g" /etc/nsswitch.conf
mkdir -p /etc/authselect
authselect select sssd with-mkhomedir --force
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
  files/etc/supervisor/conf.d/sssd.conf \
  /etc/supervisor/conf.d/
COPY files/usr/local/bin/fakesystemd.sh /usr/local/bin/
COPY files/usr/local/bin/slurmd-entrypoint.sh /usr/local/bin/entrypoint.sh

ENV PATH=/opt/amazon/${OMPI_DEFAULT}/bin:/opt/amazon/efa/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/${OMPI_DEFAULT}/lib:/opt/amazon/pmix/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/lib
ENV OPAL_PREFIX=/opt/amazon/${OMPI_DEFAULT}
ENV FI_EFA_USE_DEVICE_RDMA=1
ENV FI_EFA_FORK_SAFE=1
ENV NCCL_SOCKET_IFNAME=^docker,lo,veth

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
apt-get -qq update
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
apt-get -qq update
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
FROM base-extra AS sackd

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

USER root
WORKDIR /tmp/

ADD https://dl.k8s.io/release/stable.txt kubernetes_stable.txt

COPY --from=build /tmp/*.deb /tmp/*.ddeb /tmp/

# Ref: https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/#install-using-native-package-management
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
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
set -xeuo pipefail
apt-get -qq update
apt-get -qq -y install --no-install-recommends --fix-broken \
  socat \
  ./slurm-smd-client_[0-9]*.deb \
  ./slurm-smd-client-dbgsym_[0-9]*.ddeb \
  ./slurm-smd-doc_[0-9]*.deb \
  ./slurm-smd-sackd_[0-9]*.deb \
  ./slurm-smd-sackd-dbgsym_[0-9]*.ddeb \
  ./slurm-smd_[0-9]*.deb \
  ./slurm-smd-dbgsym_[0-9]*.ddeb
rm *.deb && rm *.ddeb
mkdir -p /run/slurm/
chown ${SLURM_USER}:${SLURM_USER} /run/slurm
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
FROM sackd AS login

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

USER root
WORKDIR /tmp/

# Ref: https://slurm.schedmd.com/quickstart_admin.html#pkg_install
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
apt-get -qq update
apt-get -qq -y install --no-install-recommends \
  socat \
  openssh-server \
  authselect sssd sssd-ad sssd-ldap libpam-sss libnss-sss
mkdir -p /etc/authselect
authselect select sssd with-mkhomedir --force
rm -f /etc/ssh/ssh_host_*
EOR

COPY --from=gdrcopy-build /usr/local/lib/libgdrapi* /usr/local/lib/
COPY --from=gdrcopy-build /usr/local/include/gdr* /usr/local/include/
RUN ldconfig

COPY files/etc/supervisor/supervisord.conf /etc/supervisor/
COPY \
  files/etc/supervisor/conf.d/fakesystemd-slurm.conf \
  files/etc/supervisor/conf.d/sackd.conf \
  files/etc/supervisor/conf.d/sshd.conf \
  files/etc/supervisor/conf.d/sssd.conf \
  /etc/supervisor/conf.d/
COPY files/usr/local/bin/fakesystemd.sh /usr/local/bin/
COPY files/usr/local/bin/login-entrypoint.sh /usr/local/bin/entrypoint.sh

EXPOSE 22/tcp
ENTRYPOINT ["entrypoint.sh"]
