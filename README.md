# HeMI: Global and Nodal Mutual Information Maximization in Heterogeneous Graphs
[Our latest version](./HeMI_2022.pdf) can be found in 'HeMI_2022.pdf'.

[Older version](https://arxiv.org/abs/2109.07008) HeMI: Multi-view Embedding in Heterogeneous Graphs (arXiv 2021).

This code supports our latest version, although some results are reproducible for the older version. Our latest version has unified some experimental observations of our older version.

## Get Started
The datasets used for the experiments can be downloaded from [this link](https://drive.google.com/file/d/1No7KkxOzdX0DVSamSsMMegcG0sqPqtZS/view?usp=share_link)
Please download them and unzip them to a folder 'data'.
You can also find our library dependencies in 'requirements.txt'. It is possible that your environment already satisfies these (or similar) dependencies, so you can try right away.

__Acknowledgements__: Datasets/Code from [HDGI](https://github.com/YuxiangRen/Heterogeneous-Deep-Graph-Infomax)  and some additional code from [Graph InfoClust](https://github.com/cmavro/Graph-InfoClust-GIC).

## Node Classification and Clustering
To run our method with all datasets, you can execute the following command. It will output node classification and node clustering results with different $\lambda$ values (see our paper). 
```
python execute_nc.py --m hemi --d acm && python execute_nc.py --m hemi --d dblp && python execute_nc.py --m hemi --d imdb
```

We also support additional methods. For example, you can use [Graph InfoClust](https://arxiv.org/abs/2009.06946) via 
```
python execute_nc.py --m gic --d acm && python execute_nc.py --m gic --d dblp && python execute_nc.py --m hemi --d imdb
```
Please navigate in `./models/' for other supported methods (DGI, DMGI, GIC, HDGI, HEMI, HGIC, MNI-DGI, SSMGRL). Some methods are not the official implementation, so you may be able to improve them.

## Link Prediction
Similarly, to reproduce the link prediction results, you can execute
```
python execute_link.py --m hemi --d acm && python execute_link.py --m hemi --d dblp && python execute_link.py --m hemi --d imdb
```
In this case, $\lambda=-1$ corresponds to having link prediction objective without HeMI. 

## Cite
If you find our code or method useful, please cite our works
```
@misc{mavromatis2022hemi,
  author={Mavromatis, Costas and Karypis, George}
  title = {Global and Nodal Mutual Information Maximization in Heterogeneous Graphs},
  howpublished = "\url{https://github.com/cmavro/HeMI/}",
  year = {2022}
}
```
or
```
@article{mavromatis2021hemi,
  title={HeMI: Multi-view Embedding in Heterogeneous Graphs},
  author={Mavromatis, Costas and Karypis, George},
  journal={arXiv preprint arXiv:2109.07008},
  year={2021}
}
```
