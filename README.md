# msd-genrecl  
  

  > please use anaconda to install the HDF5   

  ---

### Environment  

###### HDF5

please check your conda package  

```Bash
conda list  

```

uninstall hdf5 if your version is 1.8.16 (default version in conda)  

```Bash
sudo conda uninstall hdf5  

```

using anaconda search and checkout available packages  

```Bash
anaconda search -t conda hdf5  

```

we perfer you install the 1.8.15 version (or 1.8.15.1 patchy version)  

```Bash
sudo conda install hdf5=1.8.15.1  

```

###### h5py  

first check the default h5py version (2.6.0 default)  


```Bash
sudo conda uninstall h5py  

```

version 2.5.0 is recommended  

```Bash
sudo conda install h5py=2.5.0  

```

now check your packages  

```Bash
conda list  

```

---


### Scripts in Python  

more details >> [h5py](http://docs.h5py.org/en/latest/)  

```Python
import h5py  

f = h5py.File('test.h5', 'r')  

```



