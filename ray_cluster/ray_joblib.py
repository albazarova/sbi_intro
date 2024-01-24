import joblib
from ray.util.joblib import register_ray
from joblib import Parallel, delayed
from operator import neg
register_ray()


print('=====================')
for i in range(1000):
    print(neg(i + 1))

print('=====================')
with joblib.parallel_backend('ray'):
    print(Parallel(n_jobs=96)(delayed(neg)(i + 1) for i in range(1000)))

