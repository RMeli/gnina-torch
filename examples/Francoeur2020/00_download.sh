#!bin/bash

wget https://bits.csb.pitt.edu/files/crossdock2020/PDBbind2016.tar.gz
wget https://bits.csb.pitt.edu/files/crossdock2020/PDBBind2016_caches.tar.gz
wget https://bits.csb.pitt.edu/files/crossdock2020/v1.0/paper_types.tar.gz

mkdir data
tar -xvf PDBbind2016.tar.gz -C data
tar -xvf PDBBind2016_caches.tar.gz -C data
tar -xvf paper_types.tar.gz -C data
