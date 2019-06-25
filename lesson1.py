from fastai import *
from fastai.vision import *
from multiprocessing import freeze_support

if __name__ == '__main__':
            freeze_support()

path  = untar_data(URLs.PETS)
print(path)
print(path.ls())

data_path = path/'images'
print(data_path)
file_names = get_image_files(data_path)
print(file_names[:5])
# pat  = r'[^/]_d.jpg$'
np.random.seed(2)
pat = re.compile(r'/([^/]+)_\d+.jpg$')

data_bunch = ImageDataBunch.from_name_re(
    data_path,file_names,pat,ds_tfms = get_transforms(),size =224, bs =16,num_workers = 0
            ).normalize(imagenet_stats)
# data_bunch.show_batch(3, figsize = (7,6))
learn = create_cnn(data_bunch,models.resnet34,metrics = error_rate)
learn.fit(4)

print("Hello FastAI")
