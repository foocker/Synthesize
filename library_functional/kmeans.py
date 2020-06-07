import torch
import time
from tqdm import tqdm


# https://www.jianshu.com/p/1c000d9296ae
class KMEANS:
    def __init__(self, n_clusters=20, max_iter=None, verbose=True,device = torch.device("cpu")):
        # 可以看到，在特征数<3000的情况下，cpu运行速度更快，但是特征数量超过3000之后，gpu的优势越来越明显。

        self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_points = x[init_row]
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        self.representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

    def representative_sample(self):
        # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
        self.representative_samples = torch.argmin(self.dists, (0))


def time_clock(matrix,device):
    a = time.time()
    k = KMEANS(max_iter=10,verbose=False,device=device)
    k.fit(matrix)
    b = time.time()
    return (b-a)/k.count


def choose_device(cuda=False):
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure()

    device = choose_device(False)

    cpu_speeds = []
    for i in tqdm([20,100,500,2000,8000,20000]):
        matrix = torch.rand((10000,i)).to(device)
        speed = time_clock(matrix,device)
        cpu_speeds.append(speed)
    l1, = plt.plot([20,100,500,2000,8000,20000],cpu_speeds,color = 'r',label = 'CPU')

    device = choose_device(True)

    gpu_speeds = []
    for i in tqdm([20, 100, 500, 2000, 8000, 20000]):
        matrix = torch.rand((10000, i)).to(device)
        speed = time_clock(matrix,device)
        gpu_speeds.append(speed)
    l2, = plt.plot([20, 100, 500, 2000, 8000, 20000], gpu_speeds, color='g',label = "GPU")



    plt.xlabel("num_features")
    plt.ylabel("speed(s/iter)")
    plt.title("Speed with cuda")
    plt.legend(handles = [l1,l2],labels = ['CPU','GPU'],loc='best')
    plt.savefig("../result/speed.jpg")
