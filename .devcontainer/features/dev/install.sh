#!/bin/sh

arch=$(uname -m)

apt-get update -y
apt-get install -y lsb-release wget software-properties-common gnupg ninja-build
apt-get -y purge --auto-remove cmake clang-* python3-lldb-*


#
# install llvm and clang 17
#
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh 17 all
rm -f llvm.sh

update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-17 100
update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-17 100
update-alternatives --install /usr/bin/lldb lldb /usr/bin/lldb-17 100

#
# install cmake
#
cmake_version=3.27.7
cmake_fn="cmake-${cmake_version}-linux-${arch}.sh"
wget https://github.com/Kitware/CMake/releases/download/v${cmake_version}/${cmake_fn}
chmod +x ${cmake_fn}
./${cmake_fn} --skip-license --prefix=/usr/local
rm -f ${cmake_fn}

#
# install watchexec
#
watchexec_version=1.23.0
watchexec_fn="watchexec-${watchexec_version}-${arch}-unknown-linux-musl.deb"
wget -nd -P /tmp/ https://github.com/watchexec/watchexec/releases/download/v${watchexec_version}/${watchexec_fn}
dpkg -i /tmp/${watchexec_fn}
rm -f /tmp/${watchexec_fn}
