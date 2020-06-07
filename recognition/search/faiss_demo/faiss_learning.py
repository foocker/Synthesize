import numpy as np
import faiss
import time


# d = 64        # dimension
# nb = 10000    # database size
# nq = 1000     # nb of queries
# np.random.seed(42)
# xb = np.random.random((nb, d)).astype('float32')
# xb[:, 0] += np.arange(nb) / 1000.
# xq = np.random.random((nq, d)).astype('float32')
# xq[:, 0] += np.arange(nq) / 1000.


# # 为向量集构建IndexFlatL2索引，它是最简单的索引类型，只执行强力L2距离搜索
# index = faiss.IndexFlatL2(d)   # build the index
# print(index.is_trained)
# index.add(xb)    # add vectors to the index
# print(index.ntotal)

# k = 30   # we want to see 4 nearest neighbors
# D, I = index.search(xb[:5], k) # sanity check
# print(I, '\n', D)    # 向量索引位置, 相似度矩阵
# D_a, I_a = index.search(xq, k)     # actual search
# print(D_a.shape, I_a.shape)
# print(I_a[:2],'\n', D_a[:2],'\n', D_a[-2:])    # neighbors of the 5 first queries


# # https://github.com/facebookresearch/faiss/wiki/Running-on-GPUs
# # 使用GPU
# # 单GPU
# res = faiss.StandardGpuResources()  # use a single GPU
# # build a flat (CPU) index
# index_flat = faiss.IndexFlatL2(d)
# # make it into a gpu index
# gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
# gpu_index_flat.add(xb)         # add vectors to the index
# print(gpu_index_flat.ntotal)

# k = 4                          # we want to see 4 nearest neighbors
# D, I = gpu_index_flat.search(xq, k)  # actual search
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])  


# https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
# Faiss building blocks: clustering, PCA, quantization
# k-means 聚类， PCA, PQ量化

# # k-means 聚类
# ncentroids = 1024
# niter = 20
# verbose = True
# n = 20000
# d = 32
# x = np.random.rand(n, d).astype('float32')
# d = x.shape[1]
# kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
# kmeans.train(x)
# # kmeans.centroids 聚类中心
# D, I = kmeans.index.search(x, 1)
# index = faiss.IndexFlatL2(d)
# index.add (x)
# D, I = index.search(kmeans.centroids, 15)  # 最近的15个聚类中心
# print(D[:5])                   # neighbors of the 5 first queries
# print(I[:5])                  # neighbors of the 5 last queries


# Let's reduce 40D vectors to 10D
# random training data 
# mt = np.random.rand(1000, 40).astype('float32')
# mat = faiss.PCAMatrix(40, 10)
# mat.train(mt)
# assert mat.is_trained
# tr = mat.apply(mt) apply_py
# # print this to show that the magnitude of tr's columns is decreasing
# print(tr ** 2).sum(0)


