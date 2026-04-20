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
# EFA_INSTALLER_FLAGS     Flags passed to ./efa_installer.sh (MUST NOT include --skip-mpi).
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
ARG EFA_INSTALLER_FLAGS="-y --skip-kmod --skip-limit-conf --no-verify"
ARG OMPI_DEFAULT=openmpi5
ARG NCCL_CUDA_SUFFIX=cuda13.2
ARG NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"
ARG SLURM_DIR=slurm

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
  procps iputils-ping ncurses-bin \
  vim less sudo unzip git \
  build-essential pkg-config patch \
  python3 python3-pip \
  kmod \
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
  cuda-minimal-build-${CUDA_TOOLKIT_VERSION}
# Fail fast if any display/kernel driver package sneaked in via recommends
# resolution (would fight host driver at runtime).
DRV_COUNT="$(dpkg -l | awk '$2 ~ /^(cuda-drivers|nvidia-driver-|nvidia-kernel-)/ { n++ } END { print n+0 }')"
: "${DRV_COUNT:?FATAL driver-count probe produced no value}"
test "${DRV_COUNT}" -eq 0 \
  || { echo "FATAL: cuda-toolkit-${CUDA_TOOLKIT_VERSION} pulled driver package(s); aborting" >&2; dpkg -l | grep -E '^ii  (cuda-drivers|nvidia-driver-|nvidia-kernel-)' >&2; exit 1; }
# Toolkit sanity: nvcc executable + NVML header + unversioned symlink.
test -x /usr/local/cuda-13.2/bin/nvcc
test -f /usr/local/cuda-13.2/include/nvml.h
test -L /usr/local/cuda
# Pinned-suffix NCCL install. apt-cache madison emits candidate lines; the
# awk picks the newest whose version string contains the CUDA suffix token.
NCCL_PKG_VER="$(apt-cache madison libnccl2 | awk -v s="${NCCL_CUDA_SUFFIX}" '$3 ~ s { print $3; exit }')"
: "${NCCL_PKG_VER:?FATAL no libnccl2 candidate with suffix ${NCCL_CUDA_SUFFIX} in apt-cache madison}"
apt-get -qq -y install --no-install-recommends \
  "libnccl2=${NCCL_PKG_VER}" \
  "libnccl-dev=${NCCL_PKG_VER}"
EOR

ENV PATH=/usr/local/cuda/bin:${PATH}

################################################################################
# base-efa — AWS EFA installer: libfabric-aws + OMPI4 + OMPI5 + aws-ofi-nccl.
#
# Plain ubuntu:24.04 has no /opt/nvidia/nvidia_entrypoint.sh marker, so the
# EFA installer runs in normal (non-NGC) mode: it installs its own bundled
# rdma-core v61, its own OMPI4 + OMPI5 trees, its own PMIx under
# /opt/amazon/pmix, and the non-NGC aws-ofi-nccl plugin. No manual rdma-core
# preinstall, no manual libpmix-aws/prrte-aws/openmpi50-aws dpkg -i, and no
# --skip-mpi flag are needed.
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
apt-get -qq update
# environment-modules is an installer prerequisite; ca-certs + curl are
# required before the tarball fetch.
apt-get -qq -y install --no-install-recommends \
  ca-certificates curl tar environment-modules
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
# on ubuntu:24.04).
RUN <<EOR
set -xeuo pipefail
test -x /opt/amazon/openmpi5/bin/mpirun
# AWS-bundled mpirun prints `mpirun (Open MPI) 5.0.9amzn1` — match on the
# close-paren + major to avoid matching other "MPI 5" substrings.
/opt/amazon/${OMPI_DEFAULT}/bin/mpirun --version | grep -E "\(Open MPI\) 5\." >/dev/null \
  || { echo "FATAL: /opt/amazon/${OMPI_DEFAULT}/bin/mpirun did not report Open MPI 5.x" >&2; exit 1; }
test -f /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so
test -f /opt/amazon/pmix/lib/libpmix.so.2
ls -l /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so /opt/amazon/pmix/lib/libpmix.so.2
EOR

################################################################################
# slurm-build — build SchedMD .debs from source against base-efa environment.
#
# Parent MUST be base-efa (NOT slurm-src): debian/rules configure flags need
# libpmix from /opt/amazon/pmix (installed via the EFA installer's libpmix-aws
# deb) and NVML dev headers from cuda-nvml-dev-13-2 in base-cuda.
#
# Patches applied to the unpacked source tree before debuild:
#   - debian/control:    adds cuda-nvml-dev-13-2 Build-Depends; drops apt libpmix-dev.
#   - debian/rules:      adds --with-nvml=/usr/local/cuda + --with-pmix=/opt/amazon/pmix.
#   - debian/changelog:  new entry ${SLURM_VERSION}-1${SLURM_DEB_PATCH_REV}.
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
  apt-utils fakeroot devscripts equivs
EOR

COPY --from=slurm-src /workspace/${SLURM_DIR} ${SLURM_DIR}
COPY patches/ /tmp/patches/

RUN <<EOR
# Apply debian/ patches before build-dep resolution so mk-build-deps sees
# the patched debian/control.
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
# base-extra — IB userspace guest components on top of slurm-build.
#
# Parent = slurm-build so the final slurmd image is a single linear chain and
# the .debs are already present on FS (no cross-stage COPY needed). Upstream
# Slinky installs openmpi-bin/openmpi-common/libopenmpi-dev here; we omit them
# because the EFA installer already provides OMPI5 at /opt/amazon/openmpi5.
# Pulling a second OMPI tree from apt would break OPAL_PREFIX resolution at
# runtime. PMIx client lib previously sourced from libpmix2 is supplied via
# the EFA-installer's PMIx tree under /opt/amazon/pmix.
FROM slurm-build AS base-extra

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
# Mellanox OFED userspace repo — guest IB components only; kernel drivers
# are provided by the host.
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
# Auth/ops tools + IB userspace diagnostics. rdma-core was already installed
# by the EFA installer in base-efa; apt will treat the apt version as
# already-satisfied-or-newer.
apt-get -qq -y install --no-install-recommends \
  openssh-server sssd sssd-ad sssd-ldap authselect libpam-sss libnss-sss \
  gawk socat \
  infiniband-diags \
  perftest
EOR

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
apt-get -qq update
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
# Runtime env:
#   PATH            prepend /opt/amazon/${OMPI_DEFAULT}/bin + /opt/amazon/efa/bin
#   LD_LIBRARY_PATH prepend EFA + OMPI5 + PMIx + ofi-nccl + CUDA
#   OPAL_PREFIX     /opt/amazon/${OMPI_DEFAULT}  (OMPI requires this to find
#                                                 its own prrte/libevent/etc.
#                                                 outside /usr/{lib,share})
#   FI_EFA_USE_DEVICE_RDMA=1        p4/p5/p6 GPUDirect RDMA
#   FI_EFA_FORK_SAFE=1              for fork-heavy apps
#   NCCL_SOCKET_IFNAME=^docker,lo,veth  exclude container virt interfaces
FROM base-extra AS slurmd

SHELL ["bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG OMPI_DEFAULT

USER root
WORKDIR /tmp/

# slurm-build .debs already live on FS via the base-efa -> slurm-build ->
# base-extra chain; copy only the subset dpkg -i consumes into /tmp/.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
apt-get -qq update
apt-get -qq -y install --no-install-recommends --fix-broken \
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
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/${OMPI_DEFAULT}/lib:/opt/amazon/pmix/lib:/opt/amazon/ofi-nccl/lib:/usr/local/cuda/lib64:/usr/local/lib
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
# sackd .debs are already on FS via the base-extra -> slurm-build chain.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked <<EOR
set -xeuo pipefail
apt-get -qq update
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
